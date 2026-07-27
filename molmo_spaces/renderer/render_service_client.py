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

import fcntl
import logging
import mmap
import os
import struct
import time

import numpy as np

log = logging.getLogger(__name__)

_MAXSTATE = 200000
_CAM_OFF, _STATE_OFF = 1040, 1200
# Protocol v1.3 capability marker. The server writes it into cam[18] when a
# slot reaches ready; the client only READS it (it packs cam[0..17] only), so
# the marker survives every render. Its absence means a pre-v1.3 server that
# would silently render a default free camera -- the 2026-07-25 defect -- so
# the client raises instead of falling back.
_GLCAM_CAP = 20260725.0
_CAP_OFF = _CAM_OFF + 18 * 8
_PIX_OFF = _STATE_OFF + _MAXSTATE * 8
_MAX_W, _MAX_H = 624, 352
SLOT_BYTES = _PIX_OFF + _MAX_W * _MAX_H * 3

TAG_ENV = "MS_RENDER_SERVICE_TAG"
SLOT_ENV = "MS_RENDER_SERVICE_SLOT"
TAGS_ENV = "MS_RENDER_SERVICE_TAGS"  # comma list: engines CLAIM a free slot


def service_enabled() -> bool:
    if os.environ.get(TAGS_ENV):
        return True
    return bool(os.environ.get(TAG_ENV)) and os.environ.get(SLOT_ENV) is not None


def claim_any_slot(nslots: int | None = None, wait_s: float = 300.0):
    """Claim a free slot on any advertised service via flock'd lockfiles.

    Zero-coordination allocation for group-wide env vars (alice engine
    processes share identical env). The claim is an flock held open for the
    claimant's lifetime: the kernel drops it on ANY process death, so engine
    crash/quarantine churn returns the slot as soon as the owner is gone
    (O_EXCL claim files leaked slots permanently — canary 8757 deadlocked
    on exactly that). Claim files are never unlinked while services run:
    unlink+recreate would let two claimants lock different inodes.
    """
    if nslots is None:
        nslots = int(os.environ.get("MS_RENDER_SERVICE_NSLOTS", "16"))
    tags = [t for t in os.environ[TAGS_ENV].split(",") if t]
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        for tag in tags:
            for i in range(nslots):
                shm = f"/dev/shm/msrender_{tag}_slot{i}"
                if not os.path.exists(shm):
                    continue
                fd = os.open(shm + ".claim", os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
                log.info("claimed render-service slot %s/%d", tag, i)
                s = RenderServiceSlot(tag=tag, slot=i)
                s._claim_fd = fd
                return s
        time.sleep(1.0)
    raise TimeoutError("no free render-service slot")


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
        # Set from the server's cam[18] marker in init_model(); False until then.
        self.glcam_capable: bool = False
        self._claim_fd: int | None = None

    def release(self) -> None:
        """Return the slot: closing the claim fd drops the flock."""
        fd, self._claim_fd = self._claim_fd, None
        try:
            if fd is not None:
                os.close(fd)
            self._m.close()
            self._f.close()
        except Exception:
            pass

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

    def init_model(self, model_path: str, timeout_s: float = 180.0) -> int:
        b = model_path.encode()
        if len(b) > 1000:
            raise ValueError("model path too long for protocol v1")
        self._m[16:16 + len(b)] = b
        self._m[16 + len(b)] = 0
        self._set_flag(1)
        t0 = time.monotonic()
        while True:
            f = self._flag()
            if f == 5:
                break
            if f == 4:
                raise RuntimeError(f"render service failed to load {model_path}")
            if f == 3:
                # Stale 'done' from the slot's previous (dead) owner: the
                # server finished an in-flight render after our init write
                # and clobbered the flag. Path bytes are untouched; re-issue.
                self._set_flag(1)
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"render service slot {self.slot}: flag={f}")
            time.sleep(0.0002)
        self.nstate = struct.unpack_from("<i", self._m, 4)[0]
        self.glcam_capable = (
            struct.unpack_from("<d", self._m, _CAP_OFF)[0] == _GLCAM_CAP
        )
        log.info(
            "render service slot %d ready (nstate=%d glcam=%s)",
            self.slot, self.nstate, self.glcam_capable,
        )
        return self.nstate

    def render_rgb(self, state: np.ndarray, cam10, width: int, height: int,
                   derived_blob: bytes | None = None, segmentation: bool = False,
                   timeout_s: float = 60.0, cam_tail=None) -> np.ndarray:
        if width * height * 3 > _MAX_W * _MAX_H * 3:
            raise ValueError(f"resolution {width}x{height} exceeds protocol buffer")
        # cam[0..17]; cam[18..19] belong to the server (capability marker).
        cam18 = (tuple(cam10)
                 + (1.0 if derived_blob else 0.0, 1.0 if segmentation else 0.0)
                 + tuple(cam_tail if cam_tail is not None else (0.0,) * 4)
                 + (0.0, 0.0))
        struct.pack_into("<18d", self._m, _CAM_OFF, *cam18)
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


def glcam_to_cam10(gl_camera, fovy: float, width: int, height: int) -> tuple:
    """Translate a resolved ``MjvGLCamera`` into protocol v1.3 mode 3.

    THE POINT OF THIS FUNCTION (2026-07-25). ``MolmoSpacesEnv._render_frame``
    builds a throwaway ``MjvCamera``, passes it to ``update()`` purely to
    satisfy the signature, and then writes the REAL pose onto
    ``scene.camera[*]`` -- which are ``MjvGLCamera`` objects carrying
    ``pos/forward/up``. ``MjvCamera`` has no such fields, so the legacy
    ``camera_to_cam10`` path serialized an untouched placeholder and every
    service render used a constant world-origin camera
    (lookat=(0,0,0), distance=2, azimuth=90, elevation=-45).

    Mode 3 therefore transports the resolved GL camera itself. Returns the
    cam[0..9] head; pair it with :func:`glcam_tail` for cam[12..15].
    """
    return (float(gl_camera.pos[0]), float(gl_camera.pos[1]), float(gl_camera.pos[2]),
            float(gl_camera.forward[0]), float(gl_camera.forward[1]),
            float(gl_camera.forward[2]),
            3.0, -1.0, float(width), float(height))


def glcam_tail(gl_camera, fovy: float) -> tuple:
    """cam[12..15] for mode 3: up3 + fovy."""
    return (float(gl_camera.up[0]), float(gl_camera.up[1]), float(gl_camera.up[2]),
            float(fovy))


def camera_to_cam10(camera, width: int, height: int) -> tuple:
    """Translate an mjvCamera into the protocol's 10-double struct.

    LEGACY (mode 0/2). Correct only when the caller genuinely carries the pose
    in the ``MjvCamera`` -- which ``_render_frame`` does not. Kept for the
    fixed-camera path and for ``MS_RENDER_SERVICE_GLCAM=0`` A/B arms.
    """
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
