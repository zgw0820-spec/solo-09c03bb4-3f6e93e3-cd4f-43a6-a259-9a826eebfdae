"""镜像层预热批次调度引擎（纯标准库）。

在一台已完成配对的作业上，针对选定的**目标构建机**建立唯一一个预热批次，
按"本轮拉取上限（字节预算）"把申请重跑的任务依申请顺序（FIFO）放行入批：

* 目标机缓存表（作业 machines.csv）中已有的层免计；
* 入批任务共用的缺层只计一次（按层并集记账，而非按任务累加）；
* 待拉层大小未知（sizes 表缺该摘要）→ 拒绝申请并列出未知摘要；
* 单个任务的待拉量本身就超预算 → 说明原因并拒绝；
* 每次申请/撤销后从队首依次放行；队首任务剩余预算不够时，后面的任务继续等；
* 撤销后按其余任务仍需要的缺层重算占用，再从队首补位；
* 重复申请沿用已有登记，不改变排队顺序。

持久化：所有登记（队列、入批名单、缺层清单、预算、目标机、层大小表、
最近一次成功结果）都原子写入 ``prewarm/state.json``（临时文件 + fsync +
os.replace）。写盘失败时旧文件保持不动，因此重启后仍可查到最近一次成功
的结果。并发安全由进程内 RLock + 作业级文件锁（flock）共同保证，
入批名单、等候顺序、缺层清单在同一临界区/同一次落盘中一起生效。
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

try:  # fcntl 仅 POSIX 提供；部署环境为 Linux 容器
    import fcntl
except ImportError:  # pragma: no cover - 非 POSIX 平台退化到仅线程锁
    fcntl = None  # type: ignore[assignment]

# 复用配对引擎里已经验证过的列名别名
from join_engine import (DIGEST_ALIASES, MACHINE_ALIASES, TASK_ID_ALIASES,
                         _norm_header, read_csv_header)

PREWARM_DIR = "prewarm"
STATE_FILE = "state.json"
SIZES_FILE = "sizes.csv"

# 字节单位（1024 进制，大小写不敏感，支持 b/kb/mb/gib 等写法）
_UNIT_FACTORS = {
    "": 1, "b": 1,
    "k": 1024, "kb": 1024, "kib": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4, "tib": 1024 ** 4,
}
_SIZE_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]*)\s*$")

# 层大小表的字节数列别名
SIZE_ALIASES = ("字节数", "字节", "大小", "层数大小", "size", "bytes",
                "byte", "layer_size", "size_bytes", "length", "len")

# 进程内串行化所有预热操作（文件锁再兜底跨进程并发）
_global_lock = threading.RLock()


class PrewarmError(Exception):
    """可预期的业务/输入错误。status 为建议的 HTTP 状态码。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
# 锁：进程内 RLock + 作业级 flock，保证并发申请/撤销的原子性
# --------------------------------------------------------------------------- #
@contextmanager
def prewarm_lock(job_dir: str):
    pdir = os.path.join(job_dir, PREWARM_DIR)
    os.makedirs(pdir, exist_ok=True)
    lock_path = os.path.join(pdir, ".lock")
    with _global_lock:
        fh = open(lock_path, "a+")
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


# --------------------------------------------------------------------------- #
# CSV / 数值解析
# --------------------------------------------------------------------------- #
def parse_byte_count(raw: str) -> int:
    """把 ``"1024"`` / ``"3 mb"`` / ``"2GiB"`` 解析为字节整数。"""
    if raw is None:
        raise PrewarmError("字节数为空")
    text = str(raw).strip().replace(",", "")
    match = _SIZE_RE.match(text)
    if not match:
        raise PrewarmError(f"无法解析字节数：{raw!r}")
    number, unit = match.group(1), match.group(2).lower()
    factor = _UNIT_FACTORS.get(unit)
    if factor is None:
        raise PrewarmError(f"未知的字节单位：{raw!r}")
    return int(number) * factor


def _resolve(header: Optional[List[str]], aliases: Tuple[str, ...],
             label: str) -> str:
    if not header:
        raise PrewarmError("CSV 缺少表头")
    normalized = {_norm_header(c): c for c in header}
    col = next((normalized[a] for a in aliases if a in normalized), None)
    if col is None:
        raise PrewarmError(f"CSV 缺少{label}列，实际表头：{header}")
    return col


def _read_two_col_csv(path: str, key_aliases: Tuple[str, ...],
                      val_aliases: Tuple[str, ...], key_label: str,
                      val_label: str) -> Tuple[str, str]:
    header = read_csv_header(path)
    return (_resolve(header, key_aliases, key_label),
            _resolve(header, val_aliases, val_label))


def load_task_layers(job_dir: str) -> Dict[str, List[str]]:
    """任务号 -> 该任务引用的层摘要列表（按文件顺序，去重在使用处处理）。"""
    path = os.path.join(job_dir, "tasks.csv")
    if not os.path.exists(path):
        raise PrewarmError("作业缺少 tasks.csv，无法读取任务清单", 409)
    digest_col, id_col = _read_two_col_csv(
        path, DIGEST_ALIASES, TASK_ID_ALIASES, "层摘要", "任务号")
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            tid = (row.get(id_col) or "").strip()
            digest = (row.get(digest_col) or "").strip()
            if not tid or not digest:
                continue
            out.setdefault(tid, []).append(digest)
    return out


def load_machine_layers(job_dir: str) -> Dict[str, List[str]]:
    """来源构建机 -> 该机已缓存的层摘要列表。"""
    path = os.path.join(job_dir, "machines.csv")
    if not os.path.exists(path):
        raise PrewarmError("作业缺少 machines.csv，无法读取目标机缓存", 409)
    digest_col, machine_col = _read_two_col_csv(
        path, DIGEST_ALIASES, MACHINE_ALIASES, "层摘要", "来源构建机")
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            machine = (row.get(machine_col) or "").strip()
            digest = (row.get(digest_col) or "").strip()
            if not machine or not digest:
                continue
            out.setdefault(machine, []).append(digest)
    return out


def parse_sizes_csv(text: str) -> Dict[str, int]:
    """解析上传的层大小 CSV（层摘要 + 字节数），返回 digest -> bytes。

    同一摘要重复出现时大小必须一致，否则视为脏数据拒绝。
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise PrewarmError("层大小 CSV 为空")
    digest_col = _resolve(header, DIGEST_ALIASES, "层摘要")
    size_col = _resolve(header, SIZE_ALIASES, "字节数")
    digest_idx, size_idx = header.index(digest_col), header.index(size_col)

    sizes: Dict[str, int] = {}
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not (c or "").strip() for c in row):
            continue
        if len(row) <= max(digest_idx, size_idx):
            raise PrewarmError(f"第 {line_no} 行列数不足：{row}")
        digest = row[digest_idx].strip()
        if not digest:
            continue  # 空摘要行跳过
        size = parse_byte_count(row[size_idx])
        if size < 0:
            raise PrewarmError(f"第 {line_no} 行字节数为负：{size}")
        if digest in sizes and sizes[digest] != size:
            raise PrewarmError(
                f"层 {digest} 的大小冲突：{sizes[digest]} 与 {size}")
        sizes[digest] = size
    if not sizes:
        raise PrewarmError("层大小 CSV 没有任何有效数据行")
    return sizes


# --------------------------------------------------------------------------- #
# 状态持久化
# --------------------------------------------------------------------------- #
def _state_path(job_dir: str) -> str:
    return os.path.join(job_dir, PREWARM_DIR, STATE_FILE)


def _atomic_write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # 原子替换：失败/中断都不会留下半截 state
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sizes-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(job_dir: str) -> Optional[dict]:
    """读取已持久化的批次状态；未建批次返回 None。"""
    path = _state_path(job_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    # 防御性补齐字段并校正入批名单：队列/预算/大小表是权威事实，
    # 入批名单随时可以由 FIFO 放行算法确定性地重算。
    state.setdefault("queue", [])
    state.setdefault("admitted", [])
    state.setdefault("sizes", {})
    state.setdefault("last_result", None)
    sweep = admit_sweep(
        state["queue"],
        load_task_layers(job_dir),
        set(load_machine_layers(job_dir).get(state["target"], [])),
        {str(k): int(v) for k, v in state["sizes"].items()},
        state["budget"],
    )
    if sweep["admitted"] != state["admitted"]:
        state["admitted"] = sweep["admitted"]
    state["sweep"] = sweep
    return state


# --------------------------------------------------------------------------- #
# 核心记账
# --------------------------------------------------------------------------- #
def missing_layers(task_id: str, task_layers: Dict[str, List[str]],
                   cached: set) -> List[str]:
    """该任务在目标机上尚缺的层（保持任务表中的出现顺序、去重）。"""
    seen, missing = set(), []
    for digest in task_layers.get(task_id, []):
        if digest not in seen:
            seen.add(digest)
            if digest not in cached:
                missing.append(digest)
    return missing


def admit_sweep(queue: List[str], task_layers: Dict[str, List[str]],
                cached: set, sizes: Dict[str, int], budget: int) -> dict:
    """从队首依次放行，返回入批名单、缺层清单与预算占用。

    队首任务放不进时立即停止（队首阻塞），其后任务一律继续等候。
    共用缺层按并集只计一次。
    """
    admitted: List[str] = []
    required: List[str] = []          # 入批任务缺层并集，保持稳定顺序
    required_set: set = set()
    covered: set = set(cached)        # 已免计/已记账的层
    used = 0
    blocked_by: Optional[str] = None
    unknown: List[str] = []

    for task_id in queue:
        missing = missing_layers(task_id, task_layers, cached)
        incremental = 0
        task_unknown: List[str] = []
        for digest in missing:
            if digest in covered:
                continue
            if digest not in sizes:
                task_unknown.append(digest)
                continue
            incremental += int(sizes[digest])
        if task_unknown:
            # 正常路径下 apply 已拦截未知层；这里防御性地阻塞队列。
            unknown = task_unknown
            blocked_by = task_id
            break
        if used + incremental > budget:
            blocked_by = task_id  # 队首剩余预算不够，后面的继续等
            break
        admitted.append(task_id)
        for digest in missing:
            if digest not in required_set and digest in sizes:
                required_set.add(digest)
                required.append(digest)
            if digest not in covered:
                covered.add(digest)
                used += int(sizes[digest])
    return {
        "admitted": admitted,
        "required_layers": required,
        "used_bytes": used,
        "budget": budget,
        "remaining_bytes": budget - used,
        "blocked_by": blocked_by,
        "unknown_layers": unknown,
    }


def _persist(job_dir: str, state: dict) -> dict:
    """落盘并回读，保证返回的就是最近一次成功持久化的结果。"""
    _atomic_write_json(_state_path(job_dir), state)
    persisted = load_state(job_dir)
    assert persisted is not None
    return persisted


# --------------------------------------------------------------------------- #
# 对外操作
# --------------------------------------------------------------------------- #
def create_batch(job_dir: str, target: str, budget_raw: str,
                 sizes_text: str) -> dict:
    """每个作业只建一批：已存在则拒绝（409）。"""
    target = (target or "").strip()
    if not target:
        raise PrewarmError("必须选择一台目标机构建机")
    try:
        budget = parse_byte_count(budget_raw)
    except PrewarmError:
        raise
    if budget < 0:
        raise PrewarmError("拉取上限不能为负数")
    sizes = parse_sizes_csv(sizes_text)
    machine_layers = load_machine_layers(job_dir)
    if target not in machine_layers:
        raise PrewarmError(
            f"目标机 {target} 不在作业缓存表中，可选：{sorted(machine_layers)}")
    with prewarm_lock(job_dir):
        if os.path.exists(_state_path(job_dir)):
            raise PrewarmError("该作业已建立预热批次，每个作业只建一批", 409)
        state = {
            "target": target,
            "budget": budget,
            "sizes": sizes,
            "queue": [],
            "admitted": [],
            "last_result": None,
        }
        # sizes.csv 与 state 一起落盘；先写 sizes，state 是批次存在的权威标志
        _atomic_write_text(os.path.join(job_dir, PREWARM_DIR, SIZES_FILE),
                           sizes_text)
        return _persist(job_dir, state)


def apply_task(job_dir: str, task_id: str) -> dict:
    """按任务号申请入批。返回 (新状态, 是否为重复申请)。"""
    task_id = (task_id or "").strip()
    if not task_id:
        raise PrewarmError("任务号不能为空")
    with prewarm_lock(job_dir):
        state = load_state(job_dir)
        if state is None:
            raise PrewarmError("尚未建立预热批次，请先选择目标机并建立批次", 409)
        task_layers = load_task_layers(job_dir)
        if task_id not in task_layers:
            raise PrewarmError(f"任务号 {task_id} 不在本作业任务表中")

        cached = set(load_machine_layers(job_dir)[state["target"]])
        sizes = {str(k): int(v) for k, v in state["sizes"].items()}

        # 1) 待拉层大小未知 → 拒绝（列出未知摘要）
        missing = missing_layers(task_id, task_layers, cached)
        unknown = [d for d in missing if d not in sizes]
        if unknown:
            state["last_result"] = {
                "op": "apply", "task_id": task_id, "kind": "rejected",
                "message": "拒绝申请：以下待拉层的大小未知，"
                           f"无法核算预算：{', '.join(unknown)}",
            }
            _persist(job_dir, state)
            raise PrewarmError(state["last_result"]["message"])

        # 2) 单个任务的待拉量本身就超预算 → 拒绝
        own_bytes = sum(int(sizes[d]) for d in missing)
        if own_bytes > state["budget"]:
            msg = (f"拒绝申请：任务 {task_id} 待拉 {own_bytes} 字节，"
                   f"已超过本轮拉取上限 {state['budget']} 字节")
            state["last_result"] = {
                "op": "apply", "task_id": task_id, "kind": "rejected",
                "message": msg,
            }
            _persist(job_dir, state)
            raise PrewarmError(msg)

        # 3) 重复申请沿用已有登记（不改变排队顺序），返回当前状态
        duplicate = task_id in state["queue"]
        if not duplicate:
            state["queue"].append(task_id)

        sweep = admit_sweep(state["queue"], task_layers, cached, sizes,
                            state["budget"])
        state["admitted"] = sweep["admitted"]
        state["last_result"] = {
            "op": "apply", "task_id": task_id,
            "kind": "duplicate" if duplicate else (
                "admitted" if task_id in sweep["admitted"] else "queued"),
            "message": (
                f"任务 {task_id} 已在登记中（重复申请，沿用原有排队位置）"
                if duplicate else
                (f"任务 {task_id} 已放行入批，本轮待拉合计 "
                 f"{sweep['used_bytes']}/{state['budget']} 字节"
                 if task_id in sweep["admitted"] else
                 f"任务 {task_id} 已登记排队，等候前面的任务释放预算")),
        }
        return _persist(job_dir, state)


def cancel_task(job_dir: str, task_id: str) -> dict:
    """撤销申请：从队列移除并按其余任务仍需的缺层重算、从队首补位。"""
    task_id = (task_id or "").strip()
    if not task_id:
        raise PrewarmError("任务号不能为空")
    with prewarm_lock(job_dir):
        state = load_state(job_dir)
        if state is None:
            raise PrewarmError("尚未建立预热批次", 409)
        existed = task_id in state["queue"]
        if existed:
            state["queue"].remove(task_id)
        task_layers = load_task_layers(job_dir)
        cached = set(load_machine_layers(job_dir)[state["target"]])
        sizes = {str(k): int(v) for k, v in state["sizes"].items()}
        sweep = admit_sweep(state["queue"], task_layers, cached, sizes,
                            state["budget"])
        state["admitted"] = sweep["admitted"]
        state["last_result"] = {
            "op": "cancel", "task_id": task_id,
            "kind": "cancelled" if existed else "noop",
            "message": (
                f"任务 {task_id} 已撤销，其余任务仍需的缺层继续占预算，"
                f"当前待拉合计 {sweep['used_bytes']}/{state['budget']} 字节"
                if existed else f"任务 {task_id} 没有有效登记，无需撤销"),
        }
        return _persist(job_dir, state)


def summary(job_dir: str) -> Optional[dict]:
    """供页面/API 使用的完整快照（未建批次返回 None）。"""
    state = load_state(job_dir)
    if state is None:
        return None
    sweep = state["sweep"]
    task_layers = load_task_layers(job_dir)
    cached = set(load_machine_layers(job_dir)[state["target"]])
    sizes = {str(k): int(v) for k, v in state["sizes"].items()}

    admitted_detail, queue_detail = [], []
    for tid in state["queue"]:
        missing = missing_layers(tid, task_layers, cached)
        own_bytes = sum(int(sizes.get(d, 0)) for d in missing)
        item = {"task_id": tid, "missing_layers": missing,
                "own_bytes": own_bytes,
                "status": "admitted" if tid in state["admitted"] else "waiting"}
        (admitted_detail if tid in state["admitted"] else queue_detail).append(item)

    layer_detail = [{
        "digest": d, "bytes": int(sizes[d]),
        "shared_by": sorted(t for t in state["admitted"]
                            if d in missing_layers(t, task_layers, cached)),
    } for d in sweep["required_layers"]]

    return {
        "exists": True,
        "target": state["target"],
        "budget": state["budget"],
        "queue": list(state["queue"]),
        "admitted": list(state["admitted"]),
        "used_bytes": sweep["used_bytes"],
        "remaining_bytes": sweep["remaining_bytes"],
        "blocked_by": sweep["blocked_by"],
        "required_layers": layer_detail,
        "admitted_tasks": admitted_detail,
        "waiting_tasks": queue_detail,
        "last_result": state.get("last_result"),
    }
