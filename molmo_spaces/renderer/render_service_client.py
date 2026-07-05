"""Client for the per-GPU filament render service (protocol v1).

Enabled when MS_RENDER_SERVICE_TAG and MS_RENDER_SERVICE_SLOT are set for the
engine process. The service (test/ai2/render_service in the ai2-mujoco tree)
owns one filament context per slot in a single process per GPU, eliminating
the cross-process Vulkan contention that capped filament at ~16 renders/s/GPU
(2026-07-05 audit: consolidation ~9-10x; pixmatch=1.000000 vs local render).

Protocol v1 slot shm layout (see render_service.cc):
  [0] int32 flag  [4] int32 nstate  [8/12] int32 w/h
  [16] char[1024] model path (.mjb or .xml)
  [1040] 10 doubles: lookat3, distance, azimuth, elevation, type,
         fixedcamid, width, height
  [1200] double state[200000] (mjSTATE_INTEGRATION)
  [pix_off] uint8 pixels
"""
from __future__ import annotations

import logging
import mmap
import os
import struct
import time

import numpy as np

log = logging.getLogger(__name__)

_MAXSTATE = 200000
_CAM_OFF, _STATE_OFF = 1040, 1200
_PIX_OFF = _STATE_OFF + _MAXSTATE * 8
_MAX_W, _MAX_H = 624, 352
SLOT_BYTES = _PIX_OFF + _MAX_W * _MAX_H * 3

TAG_ENV = "MS_RENDER_SERVICE_TAG"
SLOT_ENV = "MS_RENDER_SERVICE_SLOT"


def service_enabled() -> bool:
    return bool(os.environ.get(TAG_ENV)) and os.environ.get(SLOT_ENV) is not None


class RenderServiceSlot:
    def __init__(self, tag: str | None = None, slot: int | None = None):
        tag = tag or os.environ[TAG_ENV]
        slot = int(os.environ[SLOT_ENV]) if slot is None else slot
        self.slot = slot
        path = f"/dev/shm/msrender_{tag}_slot{slot}"
        deadline = time.monotonic() + 120
        while not (os.path.exists(path) and os.path.getsize(path) >= SLOT_BYTES):
            if time.monotonic() > deadline:
                raise TimeoutError(f"render service slot shm missing: {path}")
            time.sleep(0.1)
        self._f = open(path, "r+b")
        self._m = mmap.mmap(self._f.fileno(), SLOT_BYTES)
        self.nstate: int | None = None

    def _flag(self) -> int:
        return struct.unpack_from("<i", self._m, 0)[0]

    def _set_flag(self, v: int) -> None:
        struct.pack_into("<i", self._m, 0, v)

    def _wait(self, want, timeout_s: float):
        t0 = time.monotonic()
        while self._flag() not in want:
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"render service slot {self.slot}: flag={self._flag()}")
            time.sleep(0.0002)
        return self._flag()

    def init_model(self, model_path: str, timeout_s: float = 900.0) -> int:
        b = model_path.encode()
        if len(b) > 1000:
            raise ValueError("model path too long for protocol v1")
        self._m[16:16 + len(b)] = b
        self._m[16 + len(b)] = 0
        self._set_flag(1)
        if self._wait((5, 4), timeout_s) == 4:
            raise RuntimeError(f"render service failed to load {model_path}")
        self.nstate = struct.unpack_from("<i", self._m, 4)[0]
        log.info("render service slot %d ready (nstate=%d)", self.slot, self.nstate)
        return self.nstate

    def render_rgb(self, state: np.ndarray, cam10, width: int, height: int,
                   derived_blob: bytes | None = None,
                   timeout_s: float = 60.0) -> np.ndarray:
        if width * height * 3 > _MAX_W * _MAX_H * 3:
            raise ValueError(f"resolution {width}x{height} exceeds protocol buffer")
        cam12 = tuple(cam10) + (1.0 if derived_blob else 0.0, 0.0)
        struct.pack_into("<12d", self._m, _CAM_OFF, *cam12)
        sb = state.tobytes()
        self._m[_STATE_OFF:_STATE_OFF + len(sb)] = sb
        if derived_blob:
            off = _STATE_OFF + len(sb)
            if off + len(derived_blob) > _PIX_OFF:
                raise ValueError("derived blob exceeds protocol buffer")
            self._m[off:off + len(derived_blob)] = derived_blob
        self._set_flag(2)
        self._wait((3,), timeout_s)
        n = width * height * 3
        return np.frombuffer(self._m[_PIX_OFF:_PIX_OFF + n], dtype=np.uint8).reshape(height, width, 3)


def camera_to_cam10(camera, width: int, height: int) -> tuple:
    """Translate an mjvCamera into the protocol's 10-double struct."""
    import mujoco as mj

    if camera.type == mj.mjtCamera.mjCAMERA_FIXED:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, float(camera.fixedcamid),
                float(width), float(height))
    return (float(camera.lookat[0]), float(camera.lookat[1]), float(camera.lookat[2]),
            float(camera.distance), float(camera.azimuth), float(camera.elevation),
            0.0, -1.0, float(width), float(height))


def derived_blob_from_data(model, data) -> bytes:
    """Post-mj_step derived kinematics, in the server's fixed order —
    exactly what mjv_updateScene reads, so the service scene is bitwise
    the client's stale-frame scene (production semantics)."""
    parts = (
        data.xpos, data.xquat, data.xmat,
        data.geom_xpos, data.geom_xmat,
        data.site_xpos, data.site_xmat,
        data.cam_xpos, data.cam_xmat,
        data.light_xpos, data.light_xdir,
    )
    return b"".join(np.ascontiguousarray(a, dtype=np.float64).tobytes() for a in parts)
