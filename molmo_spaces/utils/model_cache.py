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
        str(robot_xml_path),
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
