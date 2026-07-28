"""Coverage for render-service slot claiming (defect B2(b)).

This area had zero tests. Everything liveness-related here runs against REAL
processes and REAL flocks: the whole defect was that a fake signal (file
existence) was mistaken for liveness, so a mocked pid or a monkeypatched
os.kill would test nothing that failed in production.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.renderer import render_service_client as rsc  # noqa: E402


def _spawn_holder(tag: str) -> subprocess.Popen:
    """A real live process carrying `tag` in its argv.

    The tag must be in argv because _pid_is_live cross-checks /proc/<pid>/
    cmdline to defeat pid reuse -- the same shape as the real service, which
    is exec'd as `render_service_v2 <tag> <nslots>`.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class RenderServiceClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.shm = Path(self._tmp.name)
        self._procs: list[subprocess.Popen] = []
        self._env_backup = dict(os.environ)
        os.environ[rsc.SHM_DIR_ENV] = str(self.shm)

    def tearDown(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        os.environ.clear()
        os.environ.update(self._env_backup)
        self._tmp.cleanup()

    # -- fixtures ---------------------------------------------------------
    def _make_slots(self, tag: str, nslots: int = 1) -> None:
        for i in range(nslots):
            p = self.shm / f"msrender_{tag}_slot{i}"
            with open(p, "wb"):
                pass
            os.truncate(p, rsc.SLOT_BYTES)  # sparse; mmap needs the full size

    def _write_info(self, tag: str, pid: int) -> None:
        (self.shm / f"msrender_{tag}_info.json").write_text(
            json.dumps({"pid": pid, "tag": tag, "nslots": 1, "protocol": "1.2"})
        )

    def _beat(self, tag: str, age_s: float = 0.0) -> None:
        p = self.shm / f"msrender_{tag}_heartbeat"
        p.write_text(f"{time.time():.3f}\n")
        if age_s:
            t = time.time() - age_s
            os.utime(p, (t, t))

    def _live_service(self, tag: str, nslots: int = 1) -> subprocess.Popen:
        proc = _spawn_holder(tag)
        self._procs.append(proc)
        self._make_slots(tag, nslots)
        self._write_info(tag, proc.pid)
        self._beat(tag)
        return proc

    # -- required cases ---------------------------------------------------
    def test_live_slot_is_not_stolen_and_is_claimable(self) -> None:
        self._live_service("live")
        os.environ[rsc.TAGS_ENV] = "live"
        state, _ = rsc._service_liveness("live")
        self.assertEqual(state, "live")
        slot = rsc.claim_any_slot(nslots=1, wait_s=5.0)
        self.addCleanup(slot.release)
        self.assertEqual(slot.slot, 0)

    def test_dead_service_is_detected_and_fails_loud(self) -> None:
        proc = _spawn_holder("dead")
        pid = proc.pid
        proc.kill()
        proc.wait()  # reap: an unreaped zombie still answers os.kill(pid, 0)
        self._make_slots("dead")
        self._write_info("dead", pid)
        self._beat("dead", age_s=5.0)  # heartbeat looks recent; pid says dead

        state, detail = rsc._service_liveness("dead")
        self.assertEqual(state, "dead", detail)

        os.environ[rsc.TAGS_ENV] = "dead"
        # Fail loud, and immediately -- not after burning wait_s.
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            rsc.claim_any_slot(nslots=1, wait_s=30.0)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn(str(pid), str(cm.exception))
        # The claim must remain untaken so a restarted service is usable.
        self.assertFalse((self.shm / "msrender_dead_slot0.claim").exists())

    def test_slow_starting_service_is_not_misjudged_as_dead(self) -> None:
        """>300s of Vulkan bring-up: slots exist, heartbeat file does NOT."""
        proc = _spawn_holder("slow")
        self._procs.append(proc)
        self._make_slots("slow")
        self._write_info("slow", proc.pid)  # written before CreateContext()
        self.assertFalse((self.shm / "msrender_slow_heartbeat").exists())

        state, detail = rsc._service_liveness("slow")
        self.assertEqual(state, "starting", detail)

        os.environ[rsc.TAGS_ENV] = "slow"
        slot = rsc.claim_any_slot(nslots=1, wait_s=5.0)  # must not raise
        self.addCleanup(slot.release)

    def test_heartbeat_older_than_any_plausible_startup_is_still_not_dead(self) -> None:
        """A 32-engine spawn exceeds 300s. An mtime rule would kill it here."""
        self._live_service("hb")
        self._beat("hb", age_s=600.0)
        state, detail = rsc._service_liveness("hb")
        self.assertEqual(state, "stalled", detail)  # warn, never condemn
        os.environ[rsc.TAGS_ENV] = "hb"
        slot = rsc.claim_any_slot(nslots=1, wait_s=5.0)
        self.addCleanup(slot.release)

    def test_two_concurrent_claimers_cannot_both_win(self) -> None:
        proc = self._live_service("race", nslots=1)
        self.assertIsNone(proc.poll())
        # Both children spin until a shared wall-clock instant, so they race
        # for real instead of being separated by interpreter-startup skew. The
        # winner then holds the flock (5s) for strictly longer than the loser
        # is willing to wait (1.5s): without that ordering the loser simply
        # retries after the winner releases and "wins" too, which is correct
        # behaviour but tests nothing about mutual exclusion.
        script = textwrap.dedent(
            """
            import sys, time
            sys.path.insert(0, sys.argv[1])
            from molmo_spaces.renderer import render_service_client as rsc
            while time.time() < float(sys.argv[2]):
                time.sleep(0.001)
            try:
                s = rsc.claim_any_slot(nslots=1, wait_s=1.5)
            except Exception as e:
                print("LOST", type(e).__name__, flush=True)
            else:
                print("WON", flush=True)
                time.sleep(5.0)
                s.release()
            """
        )
        env = dict(os.environ)
        env[rsc.TAGS_ENV] = "race"
        env[rsc.SHM_DIR_ENV] = str(self.shm)
        gun = time.time() + 2.0
        kids = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(REPO_ROOT), str(gun)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env,
            )
            for _ in range(2)
        ]
        outs = [k.communicate(timeout=60)[0] for k in kids]
        self.assertEqual(sum("WON" in o for o in outs), 1, f"got {outs}")
        self.assertEqual(sum("LOST" in o for o in outs), 1, f"got {outs}")

    # -- supporting invariants --------------------------------------------
    def test_pid_reuse_does_not_read_as_live(self) -> None:
        """A recycled pid running something else is dead, not live."""
        other = _spawn_holder("some-other-tag")
        self._procs.append(other)
        self._make_slots("reused")
        self._write_info("reused", other.pid)
        self._beat("reused")
        state, detail = rsc._service_liveness("reused")
        self.assertEqual(state, "dead", detail)

    def test_pre_v2_server_without_info_json_keeps_working(self) -> None:
        """render_service.cc (v1) writes no info.json; must not hard-fail."""
        self._make_slots("v1")
        state, _ = rsc._service_liveness("v1")
        self.assertEqual(state, "unknown")
        os.environ[rsc.TAGS_ENV] = "v1"
        slot = rsc.claim_any_slot(nslots=1, wait_s=5.0)
        self.addCleanup(slot.release)

    def test_torn_info_json_read_is_not_death(self) -> None:
        """fopen("w")+fprintf is not atomic; a partial read must not condemn."""
        self._live_service("torn")
        (self.shm / "msrender_torn_info.json").write_text('{"pid":')
        state, _ = rsc._service_liveness("torn")
        self.assertEqual(state, "unknown")

    def test_one_dead_tag_does_not_block_a_live_one(self) -> None:
        gone = _spawn_holder("corpse")
        pid = gone.pid
        gone.kill()
        gone.wait()
        self._make_slots("corpse")
        self._write_info("corpse", pid)
        self._live_service("good")
        os.environ[rsc.TAGS_ENV] = "corpse,good"
        slot = rsc.claim_any_slot(nslots=1, wait_s=10.0)
        self.addCleanup(slot.release)
        self.assertEqual(slot.tag, "good")


if __name__ == "__main__":
    unittest.main()
