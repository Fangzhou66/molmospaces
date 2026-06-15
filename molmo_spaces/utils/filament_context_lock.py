import logging
import os
import time
from contextlib import contextmanager
from typing import Any


log = logging.getLogger(__name__)

_FILAMENT_CONTEXT_LOCK_PATH = "/tmp/alice_molmospaces_filament_reset.lock"
# Real B1 rollout rig on GPU0-3 (model GPU0, 48 Filament engines on GPU1-3):
# K=1 11.0 steps/s, K=2 14.4 steps/s, K=3 11.9 steps/s. K=2 overlaps
# lock-free CPU asset prep with the driver write-lock phase without over-contending.
_FILAMENT_CONTEXT_CONCURRENCY_DEFAULT = 2
_FILAMENT_CONTEXT_LOCK_TIMEOUT_S = 240.0


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        log.warning("%s=%r is <1; using %d", name, raw, default)
        return default
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a float; using %.1f", name, raw, default)
        return default
    if value <= 0:
        log.warning("%s=%r is <=0; using %.1f", name, raw, default)
        return default
    return value


@contextmanager
def filament_context_creation_lock(label: str = "mjr_context"):
    """Limit concurrent Filament MjrContext creation across processes.

    This intentionally uses the same env vars and lock-file naming as Alice's
    reset limiter, but scopes the token to the actual Vulkan/Filament context
    creation point. `flock` auto-releases on process death/SIGKILL.
    """
    scope = os.environ.get("ALICE_MS_FIL_LOCK_SCOPE", "context").lower()
    if scope not in ("context", "mjr_context", "narrow"):
        yield {
            "enabled": False,
            "waited_s": 0.0,
            "hold_s": 0.0,
            "slot": None,
            "slots": 0,
            "gpu": "disabled",
            "path": "",
            "label": label,
        }
        return

    import fcntl

    lock_path = os.environ.get(
        "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK",
        _FILAMENT_CONTEXT_LOCK_PATH,
    )
    if os.environ.get("ALICE_MS_FIL_LOCK_SHARD", "1") == "1":
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "na").split(",")[0]
        lock_path = f"{lock_path}.gpu{gpu or 'na'}"
    else:
        gpu = "global"

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    slots = _positive_int_env(
        "ALICE_MS_FIL_RESET_CONCURRENCY",
        _FILAMENT_CONTEXT_CONCURRENCY_DEFAULT,
    )
    timeout_s = _positive_float_env(
        "ALICE_MS_FIL_LOCK_TIMEOUT_S",
        _FILAMENT_CONTEXT_LOCK_TIMEOUT_S,
    )
    paths = [lock_path] if slots == 1 else [f"{lock_path}.slot{i}" for i in range(slots)]
    files = [open(path, "a+", encoding="utf-8") for path in paths]

    start = time.monotonic()
    acquired_file = None
    acquired_slot = None
    hold_start = None
    next_log = start + 30.0
    order_start = os.getpid() % slots
    info: dict[str, Any] = {
        "enabled": True,
        "waited_s": 0.0,
        "hold_s": 0.0,
        "slot": None,
        "slots": slots,
        "gpu": gpu,
        "path": lock_path,
        "label": label,
    }

    try:
        while acquired_file is None:
            for offset in range(slots):
                slot = (order_start + offset) % slots
                lock_file = files[slot]
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                acquired_file = lock_file
                acquired_slot = slot
                break

            if acquired_file is not None:
                break

            now = time.monotonic()
            waited = now - start
            if waited >= timeout_s:
                raise TimeoutError(
                    "Timed out waiting for MolmoSpaces filament context token "
                    f"after {waited:.1f}s "
                    f"(pid={os.getpid()} gpu={gpu} slots={slots} base={lock_path})"
                )
            if now >= next_log:
                log.info(
                    "MolmoSpacesEnv: still waiting %.1fs for filament context "
                    "token (pid=%d gpu=%s slots=%d label=%s)",
                    waited,
                    os.getpid(),
                    gpu,
                    slots,
                    label,
                )
                next_log = now + 30.0
            time.sleep(0.05)

        waited = time.monotonic() - start
        hold_start = time.monotonic()
        acquired_file.seek(0)
        acquired_file.truncate()
        acquired_file.write(
            f"pid={os.getpid()} gpu={gpu} label={label} acquired_at={time.time():.3f}\n"
        )
        acquired_file.flush()
        info.update({"waited_s": waited, "slot": acquired_slot})
        if waited > 1.0:
            log.info(
                "MolmoSpacesEnv: waited %.1fs for filament context token "
                "(pid=%d gpu=%s slot=%s/%d label=%s)",
                waited,
                os.getpid(),
                gpu,
                acquired_slot,
                slots,
                label,
            )
        try:
            yield info
        finally:
            info["hold_s"] = time.monotonic() - hold_start if hold_start is not None else 0.0
            fcntl.flock(acquired_file.fileno(), fcntl.LOCK_UN)
    finally:
        for lock_file in files:
            lock_file.close()
