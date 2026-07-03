import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from molmo_spaces.utils.filament_context_lock import (
    filament_context_creation_lock,
    filament_context_free_lock,
    flush_filament_context_frees,
    pending_filament_context_frees,
    resolve_filament_lock_namespace,
    schedule_filament_context_free,
)


class FilamentContextLockTests(unittest.TestCase):
    def tearDown(self) -> None:
        flush_filament_context_frees(raise_errors=False)

    def test_resolver_uses_slot_zero_for_k1_under_drain_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_LOCK_SCOPE": "context",
                "ALICE_MS_FIL_LOCK_SHARD": "1",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
                "CUDA_VISIBLE_DEVICES": "6",
            },
            clear=True,
        ):
            ns = resolve_filament_lock_namespace()

        self.assertTrue(ns["enabled"])
        self.assertEqual(ns["base_path"], f"{tmp}/fil.lock.gpu6")
        self.assertEqual(ns["slot_paths"], (f"{tmp}/fil.lock.gpu6.slot0",))
        self.assertEqual(ns["drain_path"], f"{tmp}/fil.lock.gpu6.drain")

    def test_drain_create_returns_namespace_for_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_FREE_DRAIN": "1",
                "ALICE_MS_FIL_LOCK_SCOPE": "context",
                "ALICE_MS_FIL_LOCK_SHARD": "1",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
                "ALICE_MS_FIL_LOCK_TIMEOUT_S": "1",
                "CUDA_VISIBLE_DEVICES": "2",
            },
            clear=True,
        ):
            with filament_context_creation_lock("test-create") as info:
                namespace = info["namespace"]
                self.assertEqual(info["op"], "create")
                self.assertEqual(info["slot"], 0)
                self.assertEqual(namespace["slot_paths"], (f"{tmp}/fil.lock.gpu2.slot0",))

            with filament_context_free_lock(namespace, "test-free") as free_info:
                self.assertEqual(free_info["op"], "free")
                self.assertEqual(free_info["slot"], "all")
                self.assertEqual(free_info["slots"], 1)

    def test_drain_mode_rejects_disabled_inner_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_FREE_DRAIN": "1",
                "ALICE_MS_FIL_LOCK_SCOPE": "disabled",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
            },
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                with filament_context_creation_lock("test-create"):
                    pass

    def test_free_uses_create_time_drain_metadata_when_env_drifts_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_FREE_DRAIN": "1",
                "ALICE_MS_FIL_LOCK_SCOPE": "context",
                "ALICE_MS_FIL_LOCK_SHARD": "1",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
                "ALICE_MS_FIL_LOCK_TIMEOUT_S": "1",
                "CUDA_VISIBLE_DEVICES": "3",
            },
            clear=True,
        ):
            with filament_context_creation_lock("test-create") as info:
                namespace = info["namespace"]

            os.environ["ALICE_MS_FIL_FREE_DRAIN"] = "0"
            with filament_context_free_lock(namespace, "test-free") as free_info:
                self.assertTrue(free_info["enabled"])
                self.assertEqual(free_info["slot"], "all")

    def test_free_uses_create_time_legacy_metadata_when_env_drifts_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_LOCK_SCOPE": "context",
                "ALICE_MS_FIL_LOCK_SHARD": "1",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
                "CUDA_VISIBLE_DEVICES": "4",
            },
            clear=True,
        ):
            with filament_context_creation_lock("test-create") as info:
                namespace = info["namespace"]

            os.environ["ALICE_MS_FIL_FREE_DRAIN"] = "1"
            with filament_context_free_lock(namespace, "test-free") as free_info:
                self.assertFalse(free_info["enabled"])

    def test_drain_free_rejects_missing_create_time_metadata(self) -> None:
        with patch.dict(os.environ, {"ALICE_MS_FIL_FREE_DRAIN": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                with filament_context_free_lock(None, "test-free"):
                    pass

    def test_deferred_free_requires_drain_namespace(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        with patch.dict(os.environ, {"ALICE_MS_FIL_DEFER_FREE": "1"}, clear=True):
            self.assertFalse(
                schedule_filament_context_free(
                    ctx, {"free_drain_enabled": False}, "test-free",
                )
            )
            self.assertFalse(ctx.freed)

    def test_legacy_async_flag_does_not_schedule_free(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        with patch.dict(os.environ, {"ALICE_MS_FIL_ASYNC_FREE": "1"}, clear=True):
            self.assertFalse(
                schedule_filament_context_free(
                    ctx, {"free_drain_enabled": True, "enabled": False}, "test-free",
                )
            )
            self.assertFalse(ctx.freed)
            self.assertEqual(pending_filament_context_frees(), 0)

    def test_deferred_free_flushes_fake_context(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        with patch.dict(
            os.environ,
            {
                "ALICE_MS_FIL_DEFER_FREE": "1",
                "ALICE_MS_FIL_DEFER_FREE_MAX_PENDING": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                schedule_filament_context_free(
                    ctx, {"free_drain_enabled": True, "enabled": False}, "test-free",
                )
            )
            self.assertIn(pending_filament_context_frees(), (0, 1))
            self.assertGreaterEqual(flush_filament_context_frees(), 0)
            self.assertTrue(ctx.freed)
            self.assertEqual(pending_filament_context_frees(), 0)

    def test_creation_lock_flushes_pending_deferred_free(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK": f"{tmp}/fil.lock",
                "ALICE_MS_FIL_DEFER_FREE": "1",
                "ALICE_MS_FIL_DEFER_FREE_MAX_PENDING": "1",
                "ALICE_MS_FIL_FREE_DRAIN": "1",
                "ALICE_MS_FIL_LOCK_SCOPE": "context",
                "ALICE_MS_FIL_LOCK_SHARD": "1",
                "ALICE_MS_FIL_RESET_CONCURRENCY": "1",
                "ALICE_MS_FIL_LOCK_TIMEOUT_S": "1",
                "CUDA_VISIBLE_DEVICES": "5",
            },
            clear=True,
        ):
            namespace = resolve_filament_lock_namespace()
            self.assertTrue(schedule_filament_context_free(ctx, namespace, "test-free"))
            with filament_context_creation_lock("test-create"):
                self.assertTrue(ctx.freed)
            self.assertEqual(pending_filament_context_frees(), 0)

    def test_deferred_free_flushes_from_owner_thread(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        owner_thread_id_holder: list[int] = []
        scheduled = threading.Event()
        flush_from_owner = threading.Event()
        worker_error: list[BaseException] = []

        def enqueue_from_worker() -> None:
            try:
                owner_thread_id_holder.append(threading.get_ident())
                self.assertTrue(
                    schedule_filament_context_free(
                        ctx,
                        {"free_drain_enabled": True, "enabled": False},
                        "test-free",
                        owner_thread_id=owner_thread_id_holder[0],
                    )
                )
                scheduled.set()
                flush_from_owner.wait(timeout=5)
                self.assertEqual(flush_filament_context_frees(), 1)
            except BaseException as exc:  # noqa: BLE001 - re-raise on main thread.
                worker_error.append(exc)
                scheduled.set()

        with patch.dict(
            os.environ,
            {
                "ALICE_MS_FIL_DEFER_FREE": "1",
                "ALICE_MS_FIL_DEFER_FREE_MAX_PENDING": "1",
            },
            clear=True,
        ):
            worker = threading.Thread(target=enqueue_from_worker)
            worker.start()
            self.assertTrue(scheduled.wait(timeout=5))
            if worker_error:
                raise worker_error[0]
            flush_from_owner.set()
            worker.join()
            if worker_error:
                raise worker_error[0]

        self.assertTrue(ctx.freed)
        self.assertEqual(pending_filament_context_frees(), 0)

    def test_deferred_free_flush_skips_other_owner_thread(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.freed = False

            def free(self) -> None:
                self.freed = True

        ctx = FakeContext()
        main_thread_id = threading.get_ident()
        owner_thread_id_holder: list[int] = []
        scheduled = threading.Event()
        flush_from_owner = threading.Event()
        worker_error: list[BaseException] = []

        def enqueue_from_worker() -> None:
            try:
                owner_thread_id_holder.append(threading.get_ident())
                self.assertTrue(
                    schedule_filament_context_free(
                        ctx,
                        {"free_drain_enabled": True, "enabled": False},
                        "test-free",
                        owner_thread_id=owner_thread_id_holder[0],
                    )
                )
                scheduled.set()
                flush_from_owner.wait(timeout=5)
                self.assertEqual(flush_filament_context_frees(), 1)
            except BaseException as exc:  # noqa: BLE001 - re-raise on main thread.
                worker_error.append(exc)
                scheduled.set()

        with patch.dict(
            os.environ,
            {
                "ALICE_MS_FIL_DEFER_FREE": "1",
                "ALICE_MS_FIL_DEFER_FREE_MAX_PENDING": "1",
            },
            clear=True,
        ):
            worker = threading.Thread(target=enqueue_from_worker)
            worker.start()
            self.assertTrue(scheduled.wait(timeout=5))
            if worker_error:
                raise worker_error[0]
            self.assertNotEqual(owner_thread_id_holder[0], main_thread_id)
            self.assertEqual(pending_filament_context_frees(), 1)
            self.assertEqual(flush_filament_context_frees(), 0)
            self.assertFalse(ctx.freed)
            self.assertEqual(pending_filament_context_frees(), 1)
            flush_from_owner.set()
            worker.join()
            if worker_error:
                raise worker_error[0]
            self.assertTrue(ctx.freed)
            self.assertEqual(pending_filament_context_frees(), 0)


if __name__ == "__main__":
    unittest.main()
