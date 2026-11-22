"""预热批次调度引擎自测（纯标准库，无需 Flask）。

覆盖：
- 目标机已有层免计、共用缺层只计一次
- FIFO 放行与队首阻塞（剩余预算不够队首时后面继续等）
- 待拉层大小未知 / 单任务超预算 → 说明原因并拒绝
- 重复申请沿用已有登记
- 撤销后其余任务仍需的共用缺层继续占预算，并从队首补位
- 写盘失败不破坏最近一次成功结果；重启后状态可查
- 多线程并发申请：队列顺序、入批名单、缺层清单一致且总字节不超预算

运行：
  python3 -I -m unittest -v tests.test_prewarm_engine
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

import prewarm as pw  # noqa: E402


def _write_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _make_job(root: str, task_rows: list, machine_rows: list) -> str:
    jdir = os.path.join(root, "job")
    os.makedirs(jdir, exist_ok=True)
    _write_csv(os.path.join(jdir, "tasks.csv"),
               ["任务号", "层摘要"], task_rows)
    _write_csv(os.path.join(jdir, "machines.csv"),
               ["来源构建机", "层摘要"], machine_rows)
    return jdir


def _sizes_text(mapping: dict) -> str:
    buf = "层摘要,字节数\n"
    for digest, size in mapping.items():
        buf += f"{digest},{size}\n"
    return buf


# 共用场景：
#   T1 需要 s(共用) + u1；T2 需要 s + u2；T3 需要 big；T4 需要 unknown
#   目标机 M1 已缓存 s —— s 对 M1 免计
def _scenario(root: str) -> str:
    tasks = [
        ("T1", "s"), ("T1", "u1"),
        ("T2", "s"), ("T2", "u2"),
        ("T3", "big"),
        ("T4", "unknown"),
    ]
    machines = [("M1", "s")]
    return _make_job(root, tasks, machines)


_SIZES = {"s": 30, "u1": 100, "u2": 200, "big": 500}


class AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.jdir = _scenario(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cached_free_shared_counted_once_and_hol_blocking(self):
        # 预算 350：s 在 M1 免计。T1 增量 100，T2 增量 200（共用免计层不重复），
        # T3 自身需 300（不超总预算），但剩余 50 不够 → 队首阻塞，T3 继续等。
        sizes = dict(_SIZES, big=300)
        pw.create_batch(self.jdir, "M1", "350", _sizes_text(sizes))
        pw.apply_task(self.jdir, "T1")
        pw.apply_task(self.jdir, "T2")
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["admitted"], ["T1", "T2"])
        self.assertEqual(snap["used_bytes"], 300)          # 100 + 200，s 免计
        digests = {l["digest"] for l in snap["required_layers"]}
        self.assertEqual(digests, {"u1", "u2"})            # s 不进待拉清单
        # 每层恰好一行，共用层只计一次
        self.assertEqual(len(snap["required_layers"]), 2)

        pw.apply_task(self.jdir, "T3")
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["admitted"], ["T1", "T2"])
        self.assertEqual(snap["queue"], ["T1", "T2", "T3"])
        self.assertEqual(snap["blocked_by"], "T3")
        self.assertEqual(snap["waiting_tasks"][0]["task_id"], "T3")

    def test_unknown_layer_size_rejected(self):
        pw.create_batch(self.jdir, "M1", "1000", _sizes_text(_SIZES))
        with self.assertRaises(pw.PrewarmError) as ctx:
            pw.apply_task(self.jdir, "T4")
        self.assertIn("大小未知", str(ctx.exception))
        self.assertIn("unknown", str(ctx.exception))
        snap = pw.summary(self.jdir)
        # 被拒绝的任务不得进入队列/入批名单
        self.assertEqual(snap["queue"], [])
        self.assertEqual(snap["admitted"], [])
        # 拒绝原因已落盘，重启后仍可查
        reloaded = pw.summary(self.jdir)
        self.assertEqual(reloaded["last_result"]["kind"], "rejected")
        self.assertIn("unknown", reloaded["last_result"]["message"])

    def test_single_task_over_budget_rejected(self):
        pw.create_batch(self.jdir, "M1", "400", _sizes_text(_SIZES))
        with self.assertRaises(pw.PrewarmError) as ctx:
            pw.apply_task(self.jdir, "T3")  # big=500 > 400
        self.assertIn("超过本轮拉取上限", str(ctx.exception))
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["queue"], [])
        self.assertEqual(snap["used_bytes"], 0)

    def test_duplicate_apply_keeps_registration(self):
        pw.create_batch(self.jdir, "M1", "350", _sizes_text(_SIZES))
        pw.apply_task(self.jdir, "T1")
        pw.apply_task(self.jdir, "T2")
        state2 = pw.apply_task(self.jdir, "T1")  # 重复申请
        self.assertEqual(state2["queue"], ["T1", "T2"])  # 顺序不变
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["last_result"]["kind"], "duplicate")
        self.assertEqual(snap["queue"].count("T1"), 1)

    def test_only_one_batch_per_job(self):
        pw.create_batch(self.jdir, "M1", "350", _sizes_text(_SIZES))
        with self.assertRaises(pw.PrewarmError) as ctx:
            pw.create_batch(self.jdir, "M1", "999", _sizes_text(_SIZES))
        self.assertEqual(ctx.exception.status, 409)

    def test_unknown_target_rejected(self):
        with self.assertRaises(pw.PrewarmError):
            pw.create_batch(self.jdir, "GHOST", "350", _sizes_text(_SIZES))


class CancelBackfillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        # T1: s+u1, T2: s+u2, T3: big；M1 无任何缓存 → s 也需拉取
        tasks = [("T1", "s"), ("T1", "u1"),
                 ("T2", "s"), ("T2", "u2"),
                 ("T3", "big")]
        machines = [("M1", "cached-x")]
        self.jdir = _make_job(self.tmp, tasks, machines)
        # s=30（共用），u1=100，u2=200，big=300；预算 400
        pw.create_batch(self.jdir, "M1", "400", _sizes_text(
            {"s": 30, "u1": 100, "u2": 200, "big": 300}))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cancel_keeps_shared_layer_then_backfills(self):
        pw.apply_task(self.jdir, "T1")   # 增量 130（s+u1）
        pw.apply_task(self.jdir, "T2")   # 增量 200（共用 s 只计一次）
        pw.apply_task(self.jdir, "T3")   # 需 300，剩 70 → 阻塞等候
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["used_bytes"], 330)
        self.assertEqual(snap["blocked_by"], "T3")
        self.assertEqual(
            {l["digest"] for l in snap["required_layers"]},
            {"s", "u1", "u2"})

        # 撤销 T1：u1 释放，但共用层 s 仍被 T2 需要 → 继续占预算
        pw.cancel_task(self.jdir, "T1")
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["admitted"], ["T2"])
        self.assertEqual(snap["used_bytes"], 230)       # s(30)+u2(200)
        self.assertIn("s", [l["digest"] for l in snap["required_layers"]])
        self.assertNotIn("u1", [l["digest"] for l in snap["required_layers"]])
        # 剩 170 仍不够 T3(300)，T3 继续等
        self.assertEqual(snap["blocked_by"], "T3")

        # 再撤销 T2：s 与 u2 都不再被需要 → 全部释放，T3 从队首补位
        pw.cancel_task(self.jdir, "T2")
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["admitted"], ["T3"])
        self.assertEqual(snap["used_bytes"], 300)
        self.assertEqual(snap["blocked_by"], None)
        self.assertEqual(
            [l["digest"] for l in snap["required_layers"]], ["big"])

    def test_cancel_unknown_task_is_noop(self):
        pw.apply_task(self.jdir, "T1")
        state = pw.cancel_task(self.jdir, "NOPE")
        self.assertEqual(state["queue"], ["T1"])
        self.assertEqual(pw.summary(self.jdir)["last_result"]["kind"], "noop")


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.jdir = _scenario(self.tmp)
        sizes = dict(_SIZES, big=300)  # T3 自身不超总预算，便于测排队/落盘
        pw.create_batch(self.jdir, "M1", "350", _sizes_text(sizes))
        pw.apply_task(self.jdir, "T1")
        pw.apply_task(self.jdir, "T2")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_restart_reloads_last_successful_result(self):
        # 模拟进程重启：丢弃任何内存态，只凭磁盘 state.json 重建视图
        snap = pw.summary(self.jdir)
        self.assertEqual(snap["admitted"], ["T1", "T2"])
        self.assertEqual(snap["used_bytes"], 300)
        self.assertEqual(snap["target"], "M1")
        self.assertEqual(snap["budget"], 350)
        self.assertEqual(snap["last_result"]["op"], "apply")

    def test_failed_write_keeps_last_good_state(self):
        before = pw.load_state(self.jdir)
        real_write = pw._atomic_write_json
        calls = {"n": 0}

        def flaky(path, data):
            calls["n"] += 1
            raise OSError("模拟磁盘写失败")

        pw._atomic_write_json = flaky
        try:
            with self.assertRaises(OSError):
                pw.apply_task(self.jdir, "T3")
        finally:
            pw._atomic_write_json = real_write

        # state.json 仍是最近一次成功的内容，T3 未登记
        after = pw.load_state(self.jdir)
        self.assertEqual(after["queue"], before["queue"])
        self.assertNotIn("T3", after["queue"])
        # 恢复写能力后新操作正常落盘
        pw.apply_task(self.jdir, "T3")
        self.assertIn("T3", pw.load_state(self.jdir)["queue"])
        self.assertGreaterEqual(calls["n"], 1)

    def test_sizes_file_persisted(self):
        self.assertTrue(os.path.exists(
            os.path.join(self.jdir, pw.PREWARM_DIR, pw.SIZES_FILE)))


class ConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        # 8 个任务，各含一个共用层 share 与一个独占层 uN
        tasks = []
        for i in range(8):
            tasks.append((f"T{i}", "share"))
            tasks.append((f"T{i}", f"u{i}"))
        machines = [("M1", "other")]  # 目标机没有 share/uN 缓存
        self.jdir = _make_job(self.tmp, tasks, machines)
        sizes = {"share": 40}
        sizes.update({f"u{i}": 100 for i in range(8)})
        # 预算 440：share 只计一次 → 最多放行 4 个任务（40 + 4*100 = 440）
        pw.create_batch(self.jdir, "M1", "440", _sizes_text(sizes))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_applies_never_exceed_budget(self):
        task_ids = [f"T{i}" for i in range(8)]

        def worker(tid: str) -> None:
            # 混合重复申请，验证幂等登记在并发下也成立
            pw.apply_task(self.jdir, tid)
            pw.apply_task(self.jdir, tid)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, task_ids))

        state = pw.load_state(self.jdir)
        snap = pw.summary(self.jdir)
        # 等候顺序是 8 个任务的某个排列，但每个任务恰好登记一次
        self.assertEqual(sorted(state["queue"]), task_ids)
        self.assertEqual(len(state["queue"]), 8)
        # 入批名单是队列前缀（FIFO 放行，队首阻塞）
        self.assertEqual(
            state["admitted"], state["queue"][:len(state["admitted"])])
        self.assertEqual(len(state["admitted"]), 4)
        # 待拉总字节永远不超预算；共用层 share 只计一次
        self.assertLessEqual(snap["used_bytes"], 440)
        self.assertEqual(snap["used_bytes"], 440)
        layer_bytes = sum(l["bytes"] for l in snap["required_layers"])
        self.assertEqual(layer_bytes, snap["used_bytes"])
        self.assertIn("share", [l["digest"] for l in snap["required_layers"]])
        # 缺层清单与入批名单完全对应（每个入批任务的缺层都被覆盖）
        covered = {l["digest"] for l in snap["required_layers"]}
        for t in snap["admitted_tasks"]:
            self.assertTrue(set(t["missing_layers"]).issubset(covered))

    def test_concurrent_apply_and_cancel(self):
        import time

        def applier() -> None:
            for i in range(80):
                try:
                    pw.apply_task(self.jdir, f"T{i % 8}")
                except pw.PrewarmError:
                    pass

        def canceller() -> None:
            deadline = time.time() + 1.0
            i = 0
            while time.time() < deadline:
                try:
                    pw.cancel_task(self.jdir, f"T{i % 8}")
                except pw.PrewarmError:
                    pass
                i += 1

        threads = [threading.Thread(target=applier) for _ in range(2)]
        threads.append(threading.Thread(target=canceller))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 任意时刻的落盘状态都必须自洽：不超预算、入批为队列前缀
        snap = pw.summary(self.jdir)
        self.assertLessEqual(snap["used_bytes"], snap["budget"])
        state = pw.load_state(self.jdir)
        self.assertEqual(
            state["admitted"], state["queue"][:len(state["admitted"])])
        # 缺层清单字节合计应与待拉总量一致（共用层只计一次）
        self.assertEqual(
            sum(l["bytes"] for l in snap["required_layers"]),
            snap["used_bytes"])


class CrossProcessLockTests(unittest.TestCase):
    """用独立子进程并发申请，验证 flock 在进程间也串行化、预算不超限。"""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        tasks = []
        for i in range(6):
            tasks.append((f"T{i}", "share"))
            tasks.append((f"T{i}", f"u{i}"))
        self.jdir = _make_job(self.tmp, tasks, [("M1", "other")])
        sizes = {"share": 20}
        sizes.update({f"u{i}": 100 for i in range(6)})
        pw.create_batch(self.jdir, "M1", "320", _sizes_text(sizes))
        # 预算 320：share 20 + 3*100 = 320 → 恰好 3 个任务入批
        self._expect_admitted = 3

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_subprocess_concurrent_applies(self):
        worker = """
import sys
sys.path.insert(0, %r)
import prewarm as pw
jdir = sys.argv[1]
for tid in sys.argv[2:]:
    try:
        pw.apply_task(jdir, tid)
    except pw.PrewarmError:
        pass
""" % os.path.join(REPO_ROOT, "app")
        procs = []
        for i in range(6):
            procs.append(subprocess.Popen(
                [sys.executable, "-I", "-c", worker, self.jdir,
                 f"T{i}", f"T{i}"],  # 同进程内重复申请，顺带测幂等
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        for p in procs:
            err = p.communicate(timeout=60)[1]
            self.assertEqual(p.returncode, 0, err.decode())

        snap = pw.summary(self.jdir)
        self.assertEqual(len(snap["queue"]), 6)
        self.assertEqual(len(snap["admitted"]), self._expect_admitted)
        self.assertLessEqual(snap["used_bytes"], snap["budget"])
        self.assertEqual(snap["used_bytes"], 320)
        self.assertEqual(
            snap["admitted"], snap["queue"][:len(snap["admitted"])])


class ParseTests(unittest.TestCase):
    def test_byte_units(self):
        self.assertEqual(pw.parse_byte_count("1024"), 1024)
        self.assertEqual(pw.parse_byte_count("2mb"), 2 * 1024 ** 2)
        self.assertEqual(pw.parse_byte_count(" 3 GiB "), 3 * 1024 ** 3)
        self.assertEqual(pw.parse_byte_count("1,000"), 1000)
        with self.assertRaises(pw.PrewarmError):
            pw.parse_byte_count("abc")
        with self.assertRaises(pw.PrewarmError):
            pw.parse_byte_count("5 xb")

    def test_sizes_csv_aliases_and_conflict(self):
        text = "digest,size\naaa,10\nbbb,2kb\n"
        sizes = pw.parse_sizes_csv(text)
        self.assertEqual(sizes, {"aaa": 10, "bbb": 2048})
        with self.assertRaises(pw.PrewarmError):
            pw.parse_sizes_csv("层摘要,字节数\naaa,10\naaa,11\n")
        with self.assertRaises(pw.PrewarmError):
            pw.parse_sizes_csv("foo,bar\n1,2\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
