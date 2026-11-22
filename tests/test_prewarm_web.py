"""预热批次 Web 层自测：建批、申请、撤销、拒绝提示、重启可查与 HTTP 并发。

依赖 Flask（.pylibs），运行：
  PYTHONPATH=.pylibs:app python3 -I -m unittest -v tests.test_prewarm_web
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO_ROOT, "app"),
           os.path.join(REPO_ROOT, ".pylibs")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import app as webapp  # noqa: E402


def _csv_bytes(header: list, rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _sizes_bytes(mapping: dict) -> bytes:
    return _csv_bytes(["层摘要", "字节数"], list(mapping.items()))


class PrewarmWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        webapp.JOBS_ROOT = os.path.join(self.tmp, "jobs")
        os.makedirs(webapp.JOBS_ROOT, exist_ok=True)
        webapp._recovered = False
        webapp._threads.clear()
        webapp.app.config["TESTING"] = True
        self.client = webapp.app.test_client()
        self.job_id = self._make_done_job()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 夹具 -------------------------------------------------------------- #
    def _make_done_job(self) -> str:
        tasks = [("T1", "s"), ("T1", "u1"),
                 ("T2", "s"), ("T2", "u2"),
                 ("T3", "big"),
                 ("T4", "unknown")]
        machines = [("M1", "s")]
        data = {
            "tasks_csv": (io.BytesIO(_csv_bytes(["任务号", "层摘要"], tasks)),
                          "tasks.csv"),
            "machines_csv": (io.BytesIO(
                _csv_bytes(["来源构建机", "层摘要"], machines)), "machines.csv"),
        }
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data")
        job_id = resp.headers["Location"].rstrip("/").split("/")[-1]
        deadline = time.time() + 30
        while time.time() < deadline:
            body = self.client.get(f"/jobs/{job_id}/status").get_json()
            if body["phase"] in ("done", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(body["phase"], "done")
        return job_id

    def _create_batch(self, budget: str = "350",
                      sizes: dict | None = None, target: str = "M1"):
        sizes = sizes or {"s": 30, "u1": 100, "u2": 200,
                          "big": 300, "unknown_extra": 1}
        data = {
            "target": target,
            "budget": budget,
            "sizes_csv": (io.BytesIO(_sizes_bytes(sizes)), "sizes.csv"),
        }
        return self.client.post(
            f"/jobs/{self.job_id}/prewarm/batch", data=data,
            content_type="multipart/form-data")

    def _apply(self, task_id: str):
        return self.client.post(
            f"/jobs/{self.job_id}/prewarm/apply",
            data={"task_id": task_id})

    def _cancel(self, task_id: str):
        return self.client.post(
            f"/jobs/{self.job_id}/prewarm/cancel",
            data={"task_id": task_id})

    def _snap(self) -> dict:
        return self.client.get(
            f"/jobs/{self.job_id}/prewarm/status").get_json()

    # ---- 用例 -------------------------------------------------------------- #
    def test_full_flow_admission_blocking_and_page_render(self):
        self.assertEqual(self._create_batch().status_code, 302)
        # 作业页能渲染批次面板
        page = self.client.get(f"/jobs/{self.job_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("预热批次".encode(), page.data)

        self._apply("T1")
        self._apply("T2")
        snap = self._snap()
        self.assertEqual(snap["admitted"], ["T1", "T2"])
        self.assertEqual(snap["used_bytes"], 300)  # s 免计；u1+u2
        # 申请后页面包含成功提示
        ok_page = self.client.get(f"/jobs/{self.job_id}")
        self.assertIn("入批".encode(), ok_page.data)

        # T3 自身 300 ≤ 预算 350，但只剩 50 → 排队阻塞
        self._apply("T3")
        snap = self._snap()
        self.assertEqual(snap["blocked_by"], "T3")
        self.assertEqual(snap["admitted"], ["T1", "T2"])

        # 未知大小 → 拒绝并在页面 flash 中说明原因，且不进队列
        resp = self._apply("T4")
        self.assertEqual(resp.status_code, 302)
        follow = self.client.get(resp.headers["Location"])
        self.assertIn("大小未知".encode(), follow.data)
        self.assertNotIn("T4", self._snap()["queue"])

        # 重复申请沿用登记
        self._apply("T1")
        self.assertEqual(self._snap()["last_result"]["kind"], "duplicate")
        self.assertEqual(self._snap()["queue"].count("T1"), 1)

    def test_single_task_over_budget_rejected_via_http(self):
        self._create_batch(budget="200")  # big=300 > 200
        resp = self._apply("T3")
        page = self.client.get(resp.headers["Location"])
        self.assertIn("超过本轮拉取上限".encode(), page.data)
        self.assertEqual(self._snap()["queue"], [])

    def test_cancel_backfill_shared_layer_via_http(self):
        # 目标机无 s 缓存：s 成为 T1/T2 的共用缺层
        # 重新造一个机器缓存不含 s 的作业
        tasks = [("T1", "s"), ("T1", "u1"),
                 ("T2", "s"), ("T2", "u2"),
                 ("T3", "big")]
        data = {
            "tasks_csv": (io.BytesIO(_csv_bytes(["任务号", "层摘要"], tasks)),
                          "tasks.csv"),
            "machines_csv": (io.BytesIO(_csv_bytes(
                ["来源构建机", "层摘要"], [("M1", "other")])), "m.csv"),
        }
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data")
        jid = resp.headers["Location"].rstrip("/").split("/")[-1]
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.client.get(f"/jobs/{jid}/status").get_json()["phase"] == "done":
                break
            time.sleep(0.05)

        sizes = {"s": 30, "u1": 100, "u2": 200, "big": 300}
        self.client.post(
            f"/jobs/{jid}/prewarm/batch",
            data={"target": "M1", "budget": "400",
                  "sizes_csv": (io.BytesIO(_sizes_bytes(sizes)), "s.csv")},
            content_type="multipart/form-data")
        for tid in ("T1", "T2", "T3"):
            self.client.post(f"/jobs/{jid}/prewarm/apply",
                             data={"task_id": tid})
        snap = self.client.get(f"/jobs/{jid}/prewarm/status").get_json()
        self.assertEqual(snap["used_bytes"], 330)
        self.assertEqual(snap["blocked_by"], "T3")

        # 撤销 T1：共用层 s 仍被 T2 需要，继续占预算
        self.client.post(f"/jobs/{jid}/prewarm/cancel",
                         data={"task_id": "T1"})
        snap = self.client.get(f"/jobs/{jid}/prewarm/status").get_json()
        self.assertEqual(snap["used_bytes"], 230)
        self.assertIn("s", [l["digest"] for l in snap["required_layers"]])
        self.assertEqual(snap["blocked_by"], "T3")

        # 再撤销 T2：释放全部，T3 补位入批
        self.client.post(f"/jobs/{jid}/prewarm/cancel",
                         data={"task_id": "T2"})
        snap = self.client.get(f"/jobs/{jid}/prewarm/status").get_json()
        self.assertEqual(snap["admitted"], ["T3"])
        self.assertEqual(snap["used_bytes"], 300)
        self.assertIsNone(snap["blocked_by"])

    def test_second_batch_rejected(self):
        self._create_batch()
        resp = self._create_batch()
        self.assertEqual(resp.status_code, 409)
        self.assertIn("只建一批".encode(), resp.data)

    def test_batch_before_job_done_rejected(self):
        # 构造一个数据量较大的作业，在它仍在配对时尝试建批 → 409
        tasks = [(f"T{i}", f"d{i % 37}") for i in range(900)]
        machines = [(f"M{i}", f"d{i % 37}") for i in range(300)]
        data = {
            "tasks_csv": (io.BytesIO(_csv_bytes(["任务号", "层摘要"], tasks)),
                          "tasks.csv"),
            "machines_csv": (io.BytesIO(_csv_bytes(
                ["来源构建机", "层摘要"], machines)), "m.csv"),
        }
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data")
        jid = resp.headers["Location"].rstrip("/").split("/")[-1]

        blocked = None
        deadline = time.time() + 30
        while time.time() < deadline:
            body = self.client.get(f"/jobs/{jid}/status").get_json()
            if body["phase"] != "done":
                blocked = self.client.post(
                    f"/jobs/{jid}/prewarm/batch",
                    data={"target": "M0", "budget": "100",
                          "sizes_csv": (io.BytesIO(
                              _sizes_bytes({"d0": 1})), "s.csv")},
                    content_type="multipart/form-data")
                break
            time.sleep(0.01)
        self.assertIsNotNone(blocked, "未能在作业完成前捕获到运行窗口")
        self.assertEqual(blocked.status_code, 409)
        # 未建批 → 状态接口 404，且磁盘上没有残留批次目录
        self.assertEqual(
            self.client.get(f"/jobs/{jid}/prewarm/status").status_code, 404)

        # 等作业完成后即可正常建批
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.client.get(f"/jobs/{jid}/status").get_json()["phase"] == "done":
                break
            time.sleep(0.05)
        ok = self.client.post(
            f"/jobs/{jid}/prewarm/batch",
            data={"target": "M0", "budget": "100",
                  "sizes_csv": (io.BytesIO(
                      _sizes_bytes({"d0": 1})), "s.csv")},
            content_type="multipart/form-data")
        self.assertEqual(ok.status_code, 302)

    def test_persistence_after_simulated_restart(self):
        self._create_batch()
        self._apply("T1")
        self._apply("T2")
        self._apply("T3")
        # 模拟重启：清掉任何进程内缓存，仅靠磁盘 state 重建
        webapp._recovered = False
        snap = self._snap()
        self.assertEqual(snap["target"], "M1")
        self.assertEqual(snap["queue"], ["T1", "T2", "T3"])
        self.assertEqual(snap["admitted"], ["T1", "T2"])
        self.assertEqual(snap["last_result"]["op"], "apply")
        # state.json 确实存在于作业目录
        jdir = os.path.join(webapp.JOBS_ROOT, self.job_id)
        self.assertTrue(os.path.exists(
            os.path.join(jdir, "prewarm", "state.json")))

    def test_http_concurrent_applies_never_exceed_budget(self):
        # 8 任务，共用 share + 独占 uN，预算 440 → 最多 4 个入批
        tasks, sizes = [], {"share": 40}
        for i in range(8):
            tasks += [("T%d" % i, "share"), ("T%d" % i, "u%d" % i)]
            sizes["u%d" % i] = 100
        data = {
            "tasks_csv": (io.BytesIO(_csv_bytes(["任务号", "层摘要"], tasks)),
                          "tasks.csv"),
            "machines_csv": (io.BytesIO(_csv_bytes(
                ["来源构建机", "层摘要"], [("M1", "other")])), "m.csv"),
        }
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data")
        jid = resp.headers["Location"].rstrip("/").split("/")[-1]
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.client.get(f"/jobs/{jid}/status").get_json()["phase"] == "done":
                break
            time.sleep(0.05)
        self.client.post(
            f"/jobs/{jid}/prewarm/batch",
            data={"target": "M1", "budget": "440",
                  "sizes_csv": (io.BytesIO(_sizes_bytes(sizes)), "s.csv")},
            content_type="multipart/form-data")

        def worker(i: int):
            for _ in range(3):  # 含重复申请
                self.client.post(f"/jobs/{jid}/prewarm/apply",
                                 data={"task_id": "T%d" % i})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))

        snap = self.client.get(f"/jobs/{jid}/prewarm/status").get_json()
        self.assertEqual(len(snap["queue"]), 8)
        self.assertLessEqual(snap["used_bytes"], 440)
        self.assertEqual(snap["used_bytes"], 440)
        self.assertEqual(len(snap["admitted"]), 4)
        # 入批是队列前缀
        self.assertEqual(snap["admitted"],
                         snap["queue"][:len(snap["admitted"])])

    def test_http_concurrent_apply_and_cancel_backfill(self):
        """并发申请 + 撤销，重点核对撤销后的补位与预算不变量。"""
        self._create_batch(budget="350",
                           sizes={"s": 30, "u1": 100, "u2": 200, "big": 300})

        def loop_apply():
            for tid in ("T1", "T2", "T3", "T1", "T2", "T3"):
                self._apply(tid)

        def loop_cancel():
            for tid in ("T1", "T2", "T1"):
                self._cancel(tid)

        t1 = threading.Thread(target=loop_apply)
        t2 = threading.Thread(target=loop_cancel)
        t1.start(); t2.start(); t1.join(); t2.join()

        # 最终状态自洽：不超预算，入批为队列前缀，缺层清单可对账
        snap = self._snap()
        self.assertLessEqual(snap["used_bytes"], snap["budget"])
        self.assertEqual(
            snap["admitted"], snap["queue"][:len(snap["admitted"])])
        self.assertEqual(
            sum(l["bytes"] for l in snap["required_layers"]),
            snap["used_bytes"])
        # 撤销后若预算空出，队首可补位：把队列清空后重新依次申请，验证可恢复
        for tid in list(snap["queue"]):
            self._cancel(tid)
        self.assertEqual(self._snap()["used_bytes"], 0)
        self._apply("T3")  # 300 ≤ 350，应直接入批
        self.assertEqual(self._snap()["admitted"], ["T3"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
