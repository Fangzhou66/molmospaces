import os
import tempfile
import unittest
from unittest.mock import patch

from molmo_spaces.utils.filament_context_lock import (
    filament_context_creation_lock,
    filament_context_free_lock,
    resolve_filament_lock_namespace,
)


class FilamentContextLockTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
