"""Compiled-MjModel disk cache (flag-gated, default OFF).

Cold filament rebuilds spend ~10-17s in spec build + spec.compile() per task
(MS_FILAMENT_SCENE_COMPILE_TIMING / reset_work decomposition, 2026-07-04).
Benchmark episodes are frozen, so the compiled model is a pure function of:
scene xml bytes, robot xml path, env-light intensity, asset blacklist content,
the episode spec (scene_modifications etc.), and the MuJoCo version. Caching
the compiled model as .mjb makes every later rebuild of the same episode a
~0.5-1s binary load instead of a full spec build+compile.

Enable with MS_MJB_CACHE_DIR=/path (shared across engines; atomic writes).
Bit-fidelity: mj_saveModel -> MjModel.from_binary_path is MuJoCo's own
lossless serialization; validate_roundtrip() asserts save(load(save(m))) is
byte-identical to save(m).
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import weakref

import mujoco as mj

log = logging.getLogger(__name__)

_CACHE_FORMAT = "1"


def cache_dir() -> str | None:
    d = os.environ.get("MS_MJB_CACHE_DIR", "").strip()
    return d or None


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "missing"
    return h.hexdigest()


def model_cache_key(
    scene_file_path: str,
    robot_xml_path: str,
    environment_light_intensity: float,
    episode_token: str,
    blacklist_token: str = "",
) -> str:
    h = hashlib.sha256()
    for part in (
        _CACHE_FORMAT,
        mj.__version__,
        _file_sha(str(scene_file_path)),
        # robot keyed by CONTENT (path alone would go stale on robot updates)
        str(robot_xml_path),
        _file_sha(str(robot_xml_path)),
        f"{environment_light_intensity:.6f}",
        episode_token,
        hashlib.sha256(blacklist_token.encode()).hexdigest() if blacklist_token else "noblacklist",
    ):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def load(key: str) -> "mj.MjModel | None":
    d = cache_dir()
    if not d:
        return None
    path = os.path.join(d, key + ".mjb")
    if not os.path.isfile(path):
        return None
    try:
        model = mj.MjModel.from_binary_path(path)
        log.info("MS_MJB_CACHE hit key=%s", key[:16])
        register_model_key(model, key)
        return model
    except Exception as exc:  # corrupt entry: ignore, rebuild
        log.warning("MS_MJB_CACHE corrupt entry %s: %s", path, exc)
        return None


def save(key: str, model: "mj.MjModel") -> None:
    d = cache_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".mjb.tmp", dir=d)
        os.close(fd)
        mj.mj_saveModel(model, tmp, None)
        os.replace(tmp, os.path.join(d, key + ".mjb"))
        log.info("MS_MJB_CACHE store key=%s", key[:16])
        register_model_key(model, key)
    except Exception as exc:  # cache is best-effort; never break the build
        log.warning("MS_MJB_CACHE store failed key=%s: %s", key[:16], exc)


def validate_roundtrip(model: "mj.MjModel", workdir: str) -> bool:
    """save(load(save(m))) must equal save(m) byte-for-byte."""
    a = os.path.join(workdir, "a.mjb")
    b = os.path.join(workdir, "b.mjb")
    mj.mj_saveModel(model, a, None)
    m2 = mj.MjModel.from_binary_path(a)
    mj.mj_saveModel(m2, b, None)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


# ---------------------------------------------------------------------------
# Settled-state cache (rides on the model cache key).
# Validated 2026-07-04: snapshot mjSTATE_INTEGRATION at step N-1, restore +
# one mj_step => ENTIRE MjData bitwise-identical to a fresh N-step settle
# (qpos/qvel/qacc/xpos/xquat/sensordata/act/warmstart), 0.008s vs 3.2-6.4s.
# ---------------------------------------------------------------------------

_MODEL_KEYS: "weakref.WeakValueDictionary[int, mj.MjModel]" = weakref.WeakValueDictionary()
_KEY_BY_ID: dict[int, str] = {}


def register_model_key(model: "mj.MjModel", key: str) -> None:
    try:
        _MODEL_KEYS[id(model)] = model
        _KEY_BY_ID[id(model)] = key
    except TypeError:  # model not weakref-able: registry disabled
        pass


def key_for_model(model: "mj.MjModel") -> "str | None":
    if _MODEL_KEYS.get(id(model)) is model:
        return _KEY_BY_ID.get(id(model))
    return None


def _settle_path(key: str, n_steps: int) -> "str | None":
    d = cache_dir()
    if not d:
        return None
    return os.path.join(d, f"{key}.settle{n_steps}.npz")


def load_settle_state(model: "mj.MjModel", n_steps: int):
    """Return the cached N-1 INTEGRATION state vector or None."""
    import numpy as np

    key = key_for_model(model)
    if key is None or n_steps < 1:
        return None
    path = _settle_path(key, n_steps)
    if not path or not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            if str(z["mj_version"]) != mj.__version__ or int(z["n_steps"]) != n_steps:
                return None
            state = z["state"]
        expect = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
        if state.shape[0] != expect:
            return None
        log.info("MS_SETTLE_CACHE hit key=%s n=%d", key[:16], n_steps)
        return state
    except Exception as exc:
        log.warning("MS_SETTLE_CACHE corrupt %s: %s", path, exc)
        return None


def save_settle_state(model: "mj.MjModel", n_steps: int, state) -> None:
    import numpy as np

    key = key_for_model(model)
    if key is None:
        return
    path = _settle_path(key, n_steps)
    if not path:
        return
    try:
        fd, tmp = tempfile.mkstemp(suffix=".tmp.npz", dir=os.path.dirname(path))
        os.close(fd)
        # np.savez APPENDS .npz unless the name already ends with it
        np.savez(tmp, state=state, n_steps=n_steps, mj_version=mj.__version__)
        os.replace(tmp, path)
        log.info("MS_SETTLE_CACHE store key=%s n=%d", key[:16], n_steps)
    except Exception as exc:
        log.warning("MS_SETTLE_CACHE store failed: %s", exc)
