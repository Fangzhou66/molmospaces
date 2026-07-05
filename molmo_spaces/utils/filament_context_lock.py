import logging
import os
import time
from contextlib import contextmanager
from typing import Any


log = logging.getLogger(__name__)

_FILAMENT_CONTEXT_LOCK_PATH = "/tmp/alice_molmospaces_filament_reset.lock"
# Historical B1 probe default. Current Alice filament rollout regimes inject
# ALICE_MS_FIL_RESET_CONCURRENCY explicitly (K=4 as of the 2026-07 speed
# posture); do not treat this package fallback as a current optimum.
_FILAMENT_CONTEXT_CONCURRENCY_DEFAULT = 2
_FILAMENT_CONTEXT_LOCK_TIMEOUT_S = 240.0
_FILAMENT_CONTEXT_ACTIVE_SCOPES = {"context", "mjr_context", "narrow"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


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


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def filament_free_drain_enabled() -> bool:
    return _flag_enabled("ALICE_MS_FIL_FREE_DRAIN")


def resolve_filament_lock_namespace() -> dict[str, Any]:
    """Resolve the shared Filament lifecycle lock namespace once.

    The returned namespace is used by both create and free when
    ALICE_MS_FIL_FREE_DRAIN=1, so free waits on the exact slot set used by the
    corresponding create even if environment variables drift later.
    """
    scope = os.environ.get("ALICE_MS_FIL_LOCK_SCOPE", "context").strip().lower()
    lock_path = os.environ.get(
        "ALICE_MOLMOSPACES_FILAMENT_RESET_LOCK",
        _FILAMENT_CONTEXT_LOCK_PATH,
    )
    if os.environ.get("ALICE_MS_FIL_LOCK_SHARD", "1") == "1":
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "na").split(",")[0]
        lock_path = f"{lock_path}.gpu{gpu or 'na'}"
    else:
        gpu = "global"

    slots = _positive_int_env(
        "ALICE_MS_FIL_RESET_CONCURRENCY",
        _FILAMENT_CONTEXT_CONCURRENCY_DEFAULT,
    )
    timeout_s = _positive_float_env(
        "ALICE_MS_FIL_LOCK_TIMEOUT_S",
        _FILAMENT_CONTEXT_LOCK_TIMEOUT_S,
    )
    return {
        "enabled": scope in _FILAMENT_CONTEXT_ACTIVE_SCOPES,
        "free_drain_enabled": True,
        "scope": scope,
        "base_path": lock_path,
        "path": lock_path,
        "gpu": gpu,
        "slots": slots,
        "slot_paths": tuple(f"{lock_path}.slot{i}" for i in range(slots)),
        "drain_path": f"{lock_path}.drain",
        "timeout_s": timeout_s,
    }


def _ensure_lock_dir(path: str) -> None:
    lock_dir = os.path.dirname(path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)


def _open_lock_file(path: str):
    _ensure_lock_dir(path)
    return open(path, "a+", encoding="utf-8")


def _lock_ex_until(lock_file, deadline: float, description: str) -> float:
    import fcntl

    start = time.monotonic()
    next_log = start + 30.0
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return time.monotonic() - start
        except BlockingIOError:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {description} after {now - start:.1f}s"
                )
            if now >= next_log:
                log.info(
                    "MolmoSpacesEnv: still waiting %.1fs for %s (pid=%d)",
                    now - start,
                    description,
                    os.getpid(),
                )
                next_log = now + 30.0
            time.sleep(0.05)


def _unlock_file(lock_file) -> None:
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _legacy_filament_context_creation_lock(label: str = "mjr_context"):
    """Limit concurrent Filament MjrContext creation across processes.

    This intentionally uses the same env vars and lock-file naming as Alice's
    reset limiter, but scopes the token to the actual Vulkan/Filament context
    creation point. `flock` auto-releases on process death/SIGKILL.
    """
    scope = os.environ.get("ALICE_MS_FIL_LOCK_SCOPE", "context").lower()
    if scope not in _FILAMENT_CONTEXT_ACTIVE_SCOPES:
        yield {
            "enabled": False,
            "op": "create",
            "waited_s": 0.0,
            "hold_s": 0.0,
            "slot": None,
            "slots": 0,
            "gpu": "disabled",
            "path": "",
            "label": label,
            "namespace": {"free_drain_enabled": False},
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
    # M2 fix: ALWAYS use .slot{i} including K=1 -> .slot0, matching the
    # resolver (slot_paths) so the legacy create path never aliases the bare
    # base lock (which would (a) fail to mutually exclude a drained-K=1 run and
    # (b) self-deadlock against Alice's outer lock on the same base path).
    paths = [f"{lock_path}.slot{i}" for i in range(slots)]
    files = [open(path, "a+", encoding="utf-8") for path in paths]

    start = time.monotonic()
    acquired_file = None
    acquired_slot = None
    hold_start = None
    next_log = start + 30.0
    order_start = os.getpid() % slots
    info: dict[str, Any] = {
        "enabled": True,
        "op": "create",
        "waited_s": 0.0,
        "hold_s": 0.0,
        "slot": None,
        "slots": slots,
        "gpu": gpu,
        "path": lock_path,
        "label": label,
        "namespace": {"free_drain_enabled": False},
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


@contextmanager
def _drained_filament_context_creation_lock(label: str = "mjr_context"):
    ns = resolve_filament_lock_namespace()
    if not ns["enabled"]:
        raise RuntimeError(
            "ALICE_MS_FIL_FREE_DRAIN=1 requires ALICE_MS_FIL_LOCK_SCOPE to be "
            f"one of {sorted(_FILAMENT_CONTEXT_ACTIVE_SCOPES)}; got {ns['scope']!r}"
        )

    import fcntl

    timeout_s = float(ns["timeout_s"])
    deadline = time.monotonic() + timeout_s
    drain_file = _open_lock_file(str(ns["drain_path"]))
    slot_files = [_open_lock_file(path) for path in ns["slot_paths"]]
    acquired_slot_file = None
    acquired_slot = None
    hold_start = None
    start = time.monotonic()
    order_start = os.getpid() % int(ns["slots"])
    info: dict[str, Any] = {
        "enabled": True,
        "op": "create",
        "waited_s": 0.0,
        "hold_s": 0.0,
        "slot": None,
        "slots": ns["slots"],
        "gpu": ns["gpu"],
        "path": ns["base_path"],
        "drain_path": ns["drain_path"],
        "label": label,
        "namespace": ns,
    }

    try:
        while acquired_slot_file is None:
            _lock_ex_until(drain_file, deadline, f"filament drain gate {ns['drain_path']}")
            drain_locked = True
            try:
                for offset in range(int(ns["slots"])):
                    slot = (order_start + offset) % int(ns["slots"])
                    lock_file = slot_files[slot]
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    acquired_slot_file = lock_file
                    acquired_slot = slot
                    break
            finally:
                if drain_locked:
                    _unlock_file(drain_file)

            if acquired_slot_file is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for MolmoSpaces filament context token "
                    f"after {time.monotonic() - start:.1f}s "
                    f"(pid={os.getpid()} gpu={ns['gpu']} slots={ns['slots']} "
                    f"base={ns['base_path']})"
                )
            time.sleep(0.05)

        waited = time.monotonic() - start
        hold_start = time.monotonic()
        acquired_slot_file.seek(0)
        acquired_slot_file.truncate()
        acquired_slot_file.write(
            f"pid={os.getpid()} gpu={ns['gpu']} label={label} acquired_at={time.time():.3f}\n"
        )
        acquired_slot_file.flush()
        info.update({"waited_s": waited, "slot": acquired_slot})
        if waited > 1.0:
            log.info(
                "MolmoSpacesEnv: waited %.1fs for filament context token "
                "(pid=%d gpu=%s slot=%s/%d label=%s drain=%s)",
                waited,
                os.getpid(),
                ns["gpu"],
                acquired_slot,
                ns["slots"],
                label,
                ns["drain_path"],
            )
        try:
            yield info
        finally:
            info["hold_s"] = time.monotonic() - hold_start if hold_start is not None else 0.0
            if acquired_slot_file is not None:
                _unlock_file(acquired_slot_file)
    finally:
        for lock_file in slot_files:
            lock_file.close()
        drain_file.close()


@contextmanager
def filament_context_creation_lock(label: str = "mjr_context"):
    if not filament_free_drain_enabled():
        with _legacy_filament_context_creation_lock(label) as info:
            yield info
        return
    with _drained_filament_context_creation_lock(label) as info:
        yield info


_FREE_SLOT_ENV = "ALICE_MS_FIL_FREE_CONCURRENCY"


def _free_concurrency() -> int:
    """K_free >= 1 paces MjrContext.free() with K dedicated slots per GPU
    WITHOUT touching create slots (unlike FREE_DRAIN which blocks creates).
    0/unset = unlimited (the 2026-07-05 free-storm regime)."""
    raw = os.environ.get(_FREE_SLOT_ENV, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("%s=%r is not an integer; ignoring", _FREE_SLOT_ENV, raw)
        return 0


def _acquire_any_slot(files, deadline: float, label: str) -> int:
    import fcntl

    while True:
        for i, f in enumerate(files):
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return i
            except BlockingIOError:
                continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"filament free-slot acquisition timed out ({label})")
        time.sleep(0.05)


@contextmanager
def filament_context_free_lock(namespace: dict[str, Any] | None, label: str = "mjr_context.free"):
    """Serialize MjrContext.free() against all in-flight context creates."""
    if namespace is not None and not namespace.get("free_drain_enabled", False):
        kfree = _free_concurrency()
        if kfree >= 1:
            # Paced free (2026-07-05 free-storm fix): concurrent 10-13s Engine
            # teardowns stampede the driver (close_ms 11->35s when caches
            # synchronize rebuilds). K_free slots serialize frees only.
            timeout_s = _positive_float_env(
                "ALICE_MS_FIL_LOCK_TIMEOUT_S", _FILAMENT_CONTEXT_LOCK_TIMEOUT_S
            )
            deadline = time.monotonic() + timeout_s
            base = str(namespace.get("path") or _FILAMENT_CONTEXT_LOCK_PATH)
            files = [_open_lock_file(f"{base}.freeslot{i}") for i in range(kfree)]
            start = time.monotonic()
            idx = _acquire_any_slot(files, deadline, label)
            waited = time.monotonic() - start
            try:
                yield {
                    "enabled": True,
                    "op": "free-paced",
                    "waited_s": waited,
                    "hold_s": 0.0,
                    "slot": idx,
                    "slots": kfree,
                    "gpu": namespace.get("gpu"),
                    "path": base,
                    "label": label,
                    "namespace": namespace,
                }
            finally:
                import fcntl

                try:
                    fcntl.flock(files[idx].fileno(), fcntl.LOCK_UN)
                finally:
                    for f in files:
                        f.close()
            return
        yield {
            "enabled": False,
            "op": "free",
            "waited_s": 0.0,
            "hold_s": 0.0,
            "slot": None,
            "slots": 0,
            "gpu": "disabled",
            "path": "",
            "label": label,
            "namespace": namespace,
        }
        return
    if namespace is None or not namespace.get("enabled"):
        if filament_free_drain_enabled():
            raise RuntimeError(
                "ALICE_MS_FIL_FREE_DRAIN=1 requires create-time Filament lock metadata "
                "before MjrContext.free(); refusing to destroy without synchronization."
            )
        yield {
            "enabled": False,
            "op": "free",
            "waited_s": 0.0,
            "hold_s": 0.0,
            "slot": None,
            "slots": 0,
            "gpu": "disabled",
            "path": "",
            "label": label,
            "namespace": namespace,
        }
        return

    timeout_s = float(namespace["timeout_s"])
    deadline = time.monotonic() + timeout_s
    drain_file = _open_lock_file(str(namespace["drain_path"]))
    slot_files = [_open_lock_file(path) for path in namespace["slot_paths"]]
    acquired_slot_files = []
    drain_locked = False
    hold_start = None
    start = time.monotonic()
    info: dict[str, Any] = {
        "enabled": True,
        "op": "free",
        "waited_s": 0.0,
        "hold_s": 0.0,
        "slot": "all",
        "slots": namespace["slots"],
        "gpu": namespace["gpu"],
        "path": namespace["base_path"],
        "drain_path": namespace["drain_path"],
        "label": label,
        "namespace": namespace,
    }

    try:
        _lock_ex_until(drain_file, deadline, f"filament drain gate {namespace['drain_path']}")
        drain_locked = True
        for lock_file in slot_files:
            _lock_ex_until(lock_file, deadline, f"filament free slot {lock_file.name}")
            acquired_slot_files.append(lock_file)
        waited = time.monotonic() - start
        hold_start = time.monotonic()
        info["waited_s"] = waited
        if waited > 1.0:
            log.info(
                "MolmoSpacesEnv: waited %.1fs for filament context free "
                "(pid=%d gpu=%s slots=%d label=%s drain=%s)",
                waited,
                os.getpid(),
                namespace["gpu"],
                namespace["slots"],
                label,
                namespace["drain_path"],
            )
        yield info
    finally:
        info["hold_s"] = time.monotonic() - hold_start if hold_start is not None else 0.0
        for lock_file in reversed(acquired_slot_files):
            _unlock_file(lock_file)
        if drain_locked:
            _unlock_file(drain_file)
        for lock_file in slot_files:
            lock_file.close()
        drain_file.close()
