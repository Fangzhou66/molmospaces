import logging
import os
import threading
import time
import atexit
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
_ASYNC_FREE_TASKS: list["_AsyncFreeTask"] = []
_ASYNC_FREE_ERRORS: list[BaseException] = []
_ASYNC_FREE_TASKS_LOCK = threading.Lock()
_ASYNC_FREE_ATEXIT_REGISTERED = False


class _AsyncFreeTask:
    def __init__(
        self,
        context: Any,
        namespace: dict[str, Any] | None,
        label: str,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.label = label
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"filament-async-free-{os.getpid()}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _run(self) -> None:
        try:
            _free_filament_context(
                self.context,
                self.namespace,
                self.label,
                async_free=True,
            )
        except BaseException as exc:  # noqa: BLE001 - propagate through flush.
            self.error = exc
            log.exception("Asynchronous Filament MjrContext.free failed")
        finally:
            self.context = None


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


def filament_async_free_enabled() -> bool:
    return _flag_enabled("ALICE_MS_FIL_ASYNC_FREE")


def _async_free_max_pending() -> int:
    return _positive_int_env("ALICE_MS_FIL_ASYNC_FREE_MAX_PENDING", 1)


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
    if filament_async_free_enabled():
        flush_filament_context_frees()
    if not filament_free_drain_enabled():
        with _legacy_filament_context_creation_lock(label) as info:
            yield info
        return
    with _drained_filament_context_creation_lock(label) as info:
        yield info


@contextmanager
def filament_context_free_lock(namespace: dict[str, Any] | None, label: str = "mjr_context.free"):
    """Serialize MjrContext.free() against all in-flight context creates."""
    if namespace is not None and not namespace.get("free_drain_enabled", False):
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


def _free_filament_context(
    context: Any,
    namespace: dict[str, Any] | None,
    label: str,
    *,
    async_free: bool,
) -> None:
    with filament_context_free_lock(namespace, label) as lock_info:
        free_t0 = time.monotonic()
        context.free()
        free_s = time.monotonic() - free_t0
    log.info(
        "MS_FILAMENT_MJR_CONTEXT_FREE_TIMING gpu=%s slots=%s "
        "wait_s=%.3f free_s=%.3f lock_hold_s=%.3f async=%d",
        lock_info.get("gpu"),
        lock_info.get("slots"),
        float(lock_info.get("waited_s") or 0.0),
        free_s,
        float(lock_info.get("hold_s") or 0.0),
        1 if async_free else 0,
    )


def _unfinished_async_tasks_locked() -> list[_AsyncFreeTask]:
    return [task for task in _ASYNC_FREE_TASKS if task.thread.is_alive()]


def _reap_async_tasks_locked() -> None:
    unfinished = []
    for task in _ASYNC_FREE_TASKS:
        if task.thread.is_alive():
            unfinished.append(task)
        elif task.error is not None:
            _ASYNC_FREE_ERRORS.append(task.error)
    _ASYNC_FREE_TASKS[:] = unfinished


def pending_filament_context_frees() -> int:
    with _ASYNC_FREE_TASKS_LOCK:
        _reap_async_tasks_locked()
        return len(_ASYNC_FREE_TASKS)


def flush_filament_context_frees(
    *,
    timeout_s: float | None = None,
    raise_errors: bool = True,
) -> int:
    """Wait for all background MjrContext.free calls in this process.

    This is the force-flush hook Alice should call before an offload boundary
    returns GPU memory to the trainer and before process shutdown.
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    flushed = 0
    errors: list[BaseException] = []
    while True:
        with _ASYNC_FREE_TASKS_LOCK:
            _reap_async_tasks_locked()
            errors.extend(_ASYNC_FREE_ERRORS)
            _ASYNC_FREE_ERRORS.clear()
            tasks = list(_ASYNC_FREE_TASKS)
        if not tasks:
            break
        for task in tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not task.join(remaining):
                raise TimeoutError(
                    "Timed out waiting for asynchronous Filament MjrContext.free"
                )
            flushed += 1
            if task.error is not None:
                errors.append(task.error)
        with _ASYNC_FREE_TASKS_LOCK:
            _ASYNC_FREE_TASKS[:] = [
                task for task in _ASYNC_FREE_TASKS if task.thread.is_alive()
            ]
    with _ASYNC_FREE_TASKS_LOCK:
        errors.extend(_ASYNC_FREE_ERRORS)
        _ASYNC_FREE_ERRORS.clear()
    if errors and raise_errors:
        raise RuntimeError(
            f"{len(errors)} asynchronous Filament MjrContext.free call(s) failed"
        ) from errors[0]
    if flushed:
        log.info("MS_FILAMENT_ASYNC_FREE_FLUSH flushed=%d", flushed)
    return flushed


def _flush_filament_context_frees_at_exit() -> None:
    try:
        flush_filament_context_frees(raise_errors=False)
    except Exception:
        log.exception("Failed to flush asynchronous Filament frees at exit")


def schedule_filament_context_free(
    context: Any,
    namespace: dict[str, Any] | None,
    label: str = "mjr_context.free",
) -> bool:
    """Schedule MjrContext.free() on a bounded background lane.

    Returns False when async free is disabled, so callers can fall back to the
    exact synchronous path. The background task still uses
    filament_context_free_lock(), preserving create/free exclusion.
    """
    if not filament_async_free_enabled():
        return False
    if namespace is None or not namespace.get("free_drain_enabled", False):
        return False

    global _ASYNC_FREE_ATEXIT_REGISTERED
    max_pending = _async_free_max_pending()
    while pending_filament_context_frees() >= max_pending:
        flush_filament_context_frees()

    task = _AsyncFreeTask(context, namespace, label)
    with _ASYNC_FREE_TASKS_LOCK:
        _ASYNC_FREE_TASKS.append(task)
        pending = len(_ASYNC_FREE_TASKS)
        if not _ASYNC_FREE_ATEXIT_REGISTERED:
            atexit.register(_flush_filament_context_frees_at_exit)
            _ASYNC_FREE_ATEXIT_REGISTERED = True
    task.start()
    log.info(
        "MS_FILAMENT_ASYNC_FREE_SCHEDULED pending=%d max_pending=%d label=%s",
        pending,
        max_pending,
        label,
    )
    return True
