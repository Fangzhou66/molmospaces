import logging
import hashlib
import os
import queue
import threading
import time
import traceback
from typing import Any

import mujoco as mj
import numpy as np

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer
from molmo_spaces.utils.filament_context_lock import (
    filament_context_creation_lock,
    filament_context_free_lock,
    schedule_filament_context_free,
)

import os as _os
# Datagen render-diet only: the upstream plain-RGB path does a redundant 2nd
# mjr_readPixels (commit 5b643a3). It is byte-identical -- one read already runs
# the full beginFrame->render->ReadColorPixels->endFrame->flushAndWait pipeline,
# so the 2nd is a wasted GPU->host readback under the contended per-GPU RM lock.
# Gated on MS_RENDER_KEEP_CAMERAS so DEFAULT / RL behaviour stays byte-identical
# to upstream (double read); only diet datagen workers drop the duplicate.
_MS_RENDER_DIET = _os.environ.get("MS_RENDER_KEEP_CAMERAS") is not None
log = logging.getLogger(__name__)

_PROCESS_TEXTURE_KEYS: set[str] = set()
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ACTOR_CLOSE_LOCK = threading.Lock()
_ACTOR_PENDING_CLOSES: list[tuple["_FilamentRendererActor", "queue.Queue[tuple[bool, Any]]"]] = []


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


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


def _actor_enabled() -> bool:
    return _flag_enabled("ALICE_MS_FIL_RENDERER_ACTOR")


def _actor_async_close_enabled() -> bool:
    return _flag_enabled("ALICE_MS_FIL_RENDERER_ACTOR_ASYNC_CLOSE")


def _actor_call_timeout_s() -> float:
    return _positive_float_env("ALICE_MS_FIL_RENDERER_ACTOR_CALL_TIMEOUT_S", 300.0)


def _actor_max_pending_close() -> int:
    return _positive_int_env("ALICE_MS_FIL_RENDERER_ACTOR_MAX_PENDING_CLOSE", 2)


def _await_actor_close(
    actor: "_FilamentRendererActor",
    result_queue: "queue.Queue[tuple[bool, Any]]",
) -> None:
    ok, result = result_queue.get(timeout=_actor_call_timeout_s())
    actor.join(timeout=_actor_call_timeout_s())
    if not ok:
        raise result


def _drain_filament_renderer_actor_closes(*, force: bool) -> int:
    flushed = 0
    with _ACTOR_CLOSE_LOCK:
        kept: list[tuple[_FilamentRendererActor, queue.Queue[tuple[bool, Any]]]] = []
        for actor, result_queue in _ACTOR_PENDING_CLOSES:
            if force:
                _await_actor_close(actor, result_queue)
                flushed += 1
                continue
            try:
                ok, result = result_queue.get_nowait()
            except queue.Empty:
                kept.append((actor, result_queue))
                continue
            actor.join(timeout=0.0)
            flushed += 1
            if not ok:
                raise result
        _ACTOR_PENDING_CLOSES[:] = kept

        while len(_ACTOR_PENDING_CLOSES) >= _actor_max_pending_close():
            actor, result_queue = _ACTOR_PENDING_CLOSES.pop(0)
            _await_actor_close(actor, result_queue)
            flushed += 1
    return flushed


def flush_filament_renderer_actors() -> int:
    """Wait for all async actor-owned renderer closes in this process."""
    return _drain_filament_renderer_actor_closes(force=True)


class _FilamentRendererActor:
    def __init__(self, renderer_kwargs: dict[str, Any]) -> None:
        _drain_filament_renderer_actor_closes(force=False)
        self._requests: queue.Queue[tuple[str, tuple[Any, ...], dict[str, Any], queue.Queue]] = (
            queue.Queue()
        )
        self._ready: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(renderer_kwargs,),
            name=f"filament-renderer-actor-{os.getpid()}",
        )
        self._thread.start()
        ok, result = self._ready.get(timeout=_actor_call_timeout_s())
        if not ok:
            self._thread.join(timeout=0.1)
            raise result

    def _run(self, renderer_kwargs: dict[str, Any]) -> None:
        renderer = None
        ready_sent = False
        result_queue: queue.Queue | None = None
        try:
            renderer = MjFilamentRenderer(_actor_inner=True, **renderer_kwargs)
            self._ready.put((True, None))
            ready_sent = True
            while True:
                op, args, kwargs, result_queue = self._requests.get()
                try:
                    if op == "__close__":
                        t0 = time.monotonic()
                        renderer.close()
                        renderer = None
                        log.info(
                            "MS_FILAMENT_RENDERER_ACTOR_CLOSE_DONE close_s=%.3f",
                            time.monotonic() - t0,
                        )
                        result_queue.put((True, None))
                        return
                    result = getattr(renderer, op)(*args, **kwargs)
                    result_queue.put((True, result))
                except BaseException as exc:  # noqa: BLE001
                    log.error(
                        "MS_FILAMENT_RENDERER_ACTOR_CALL_FAILED op=%s\n%s",
                        op,
                        traceback.format_exc(),
                    )
                    result_queue.put((False, exc))
        except BaseException as exc:  # noqa: BLE001
            log.error("MS_FILAMENT_RENDERER_ACTOR_FAILED\n%s", traceback.format_exc())
            if not ready_sent:
                self._ready.put((False, exc))
            elif result_queue is not None:
                result_queue.put((False, exc))
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except BaseException:  # noqa: BLE001
                    log.error(
                        "MS_FILAMENT_RENDERER_ACTOR_FINAL_CLOSE_FAILED\n%s",
                        traceback.format_exc(),
                    )

    def call(self, op: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError(f"Filament renderer actor is closed; cannot call {op}")
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._requests.put((op, args, kwargs, result_queue))
        ok, result = result_queue.get(timeout=_actor_call_timeout_s())
        if not ok:
            raise result
        return result

    def close(self, *, wait: bool) -> None:
        if self._closed:
            return
        self._closed = True
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        self._requests.put(("__close__", (), {}, result_queue))
        if wait:
            _await_actor_close(self, result_queue)
            return
        with _ACTOR_CLOSE_LOCK:
            _ACTOR_PENDING_CLOSES.append((self, result_queue))
        log.info(
            "MS_FILAMENT_RENDERER_ACTOR_CLOSE_SCHEDULED pending=%d",
            len(_ACTOR_PENDING_CLOSES),
        )

    def join(self, *, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)


def _model_cstring(chars, start: int) -> str:
    if start < 0:
        return ""
    try:
        raw = np.asarray(chars).tobytes()[start:]
    except Exception:
        raw = bytes(chars)[start:]
    end = raw.find(b"\x00")
    if end >= 0:
        raw = raw[:end]
    return raw.decode("utf-8", errors="replace")


def _texture_key(model: mj.MjModel, tex_id: int) -> tuple[str, int]:
    height = int(model.tex_height[tex_id])
    width = int(model.tex_width[tex_id])
    nchannel = int(model.tex_nchannel[tex_id])
    nbytes = height * width * nchannel

    path = _model_cstring(model.paths, int(model.tex_pathadr[tex_id]))
    if path:
        return f"path:{path}", nbytes

    adr = int(model.tex_adr[tex_id])
    tex_data = np.asarray(model.tex_data, dtype=np.uint8)[adr : adr + nbytes]
    digest = hashlib.blake2b(tex_data, digest_size=16).hexdigest()
    return (
        "data:"
        f"type={int(model.tex_type[tex_id])}:"
        f"colorspace={int(model.tex_colorspace[tex_id])}:"
        f"shape={height}x{width}x{nchannel}:"
        f"{digest}",
        nbytes,
    )


def _log_texture_cache_potential(model: mj.MjModel) -> None:
    if os.environ.get("ALICE_MS_FIL_TEXTURE_CACHE_LOG", "1").lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return

    keys = []
    total_bytes = 0
    for tex_id in range(int(model.ntex)):
        key, nbytes = _texture_key(model, tex_id)
        keys.append(key)
        total_bytes += nbytes

    unique_keys = set(keys)
    hits = sum(1 for key in keys if key in _PROCESS_TEXTURE_KEYS)
    misses = len(unique_keys - _PROCESS_TEXTURE_KEYS)
    _PROCESS_TEXTURE_KEYS.update(unique_keys)
    hit_rate = hits / len(keys) if keys else 0.0
    log.info(
        "MS_FILAMENT_TEXTURE_CACHE_POTENTIAL pid=%d ntex=%d unique_in_model=%d "
        "duplicate_in_model=%d process_seen_hits=%d process_new_unique=%d "
        "process_seen_total_unique=%d hit_rate=%.3f texture_mb=%.1f",
        os.getpid(),
        int(model.ntex),
        len(unique_keys),
        len(keys) - len(unique_keys),
        hits,
        misses,
        len(_PROCESS_TEXTURE_KEYS),
        hit_rate,
        total_bytes / (1024 * 1024),
    )


def prepare_locals_for_super(
    local_vars, args_name="args", kwargs_name="kwargs", ignore_kwargs=False
):
    assert args_name not in local_vars, f"`prepare_locals_for_super` does not support {args_name}."
    new_locals = {k: v for k, v in local_vars.items() if k != "self" and "__" not in k}
    if kwargs_name in new_locals:
        if ignore_kwargs:
            new_locals.pop(kwargs_name)
        else:
            kwargs = new_locals.pop(kwargs_name)
            kwargs.update(new_locals)
            new_locals = kwargs
    return new_locals


class MjFilamentRenderer(MjAbstractRenderer):
    def __init__(
        self,
        model_bindings: MjModelBindings = None,
        device_id: int | None = None,
        height: int = 720,
        width: int = 1280,
        max_geom: int = 10000,
        model: mj.MjModel | None = None,
        _actor_inner: bool = False,
        **kwargs: Any,
    ) -> None:
        if _actor_enabled() and not _actor_inner:
            super().__init__(**prepare_locals_for_super(locals()))
            renderer_kwargs = prepare_locals_for_super(
                locals(),
                ignore_kwargs=True,
            )
            renderer_kwargs.pop("_actor_inner", None)
            self._actor = _FilamentRendererActor(renderer_kwargs)
            self._actor_proxy = True
            self._closed = False
            return

        self._actor_proxy = False
        self._actor = None
        del _actor_inner
        assert model_bindings is not None or model is not None, (
            "model_bindings or model must be provided"
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self._width = width
        self._height = height

        if model_bindings is not None and model is not None:
            assert model_bindings.model == model, "model_bindings and model must be the same"
        model = model_bindings.model if model_bindings is not None else model
        self._model = model

        self._scene = mj.MjvScene(model=model, maxgeom=max_geom)
        self._scene_option = mj.MjvOption()

        # Turn off site rendering
        self._scene_option.sitegroup *= 0

        # Enable shadow rendering by default (shadows are controlled by lights with castshadow enabled)
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        _log_texture_cache_potential(model)
        self._filament_context_thread_id = threading.get_ident()
        with filament_context_creation_lock("MjFilamentRenderer.MjrContext") as lock_info:
            context_t0 = time.monotonic()
            self._mjr_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
            context_s = time.monotonic() - context_t0
        self._filament_lock_namespace = lock_info.get("namespace")
        log.info(
            "MS_FILAMENT_MJR_CONTEXT_TIMING gpu=%s slots=%s slot=%s "
            "wait_s=%.3f context_s=%.3f lock_hold_s=%.3f "
            "ngeom=%d nmesh=%d ntex=%d",
            lock_info.get("gpu"),
            lock_info.get("slots"),
            lock_info.get("slot"),
            float(lock_info.get("waited_s") or 0.0),
            context_s,
            float(lock_info.get("hold_s") or 0.0),
            int(model.ngeom),
            int(model.nmesh),
            int(model.ntex),
        )
        # mj.mjr_resizeOffscreen(width, height, self._mjr_context)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
        self._mjr_context.readDepthMap = mj.mjtDepthMap.mjDEPTH_ZEROFAR

        # Default render flags.
        self._depth_rendering = False
        self._segmentation_rendering = False

        # Track if textures need to be uploaded (set to True when textures are modified)
        # NOTE: We start with False because textures are loaded from model at MjrContext creation
        # We only need to upload if textures are modified AFTER renderer initialization
        self._textures_need_upload = False

    @property
    def scene(self) -> mj.MjvScene:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("scene")
        return self._scene

    @property
    def height(self):
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("height")
        return self._height

    @property
    def width(self):
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("width")
        return self._width

    def enable_depth_rendering(self) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("enable_depth_rendering")
        self._segmentation_rendering = False
        self._depth_rendering = True

    def disable_depth_rendering(self) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("disable_depth_rendering")
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("enable_segmentation_rendering")
        self._segmentation_rendering = True
        self._depth_rendering = False

    def disable_segmentation_rendering(self) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("disable_segmentation_rendering")
        self._segmentation_rendering = False

    def geomid_to_bodyid(self, geomid):
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("geomid_to_bodyid", geomid)
        return self.model.geom_bodyid[geomid]

    def set_scene_camera_pose(
        self,
        pos: np.ndarray,
        forward: np.ndarray,
        up: np.ndarray,
    ) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("set_scene_camera_pose", pos, forward, up)
        for camera in self._scene.camera:
            camera.pos = pos
            camera.forward = forward
            camera.up = up

    def set_scene_camera_orthographic_frustum(
        self,
        *,
        frustum_bottom: float,
        frustum_top: float,
    ) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call(
                "set_scene_camera_orthographic_frustum",
                frustum_bottom=frustum_bottom,
                frustum_top=frustum_top,
            )
        for camera in self._scene.camera:
            camera.orthographic = 1
            camera.frustum_bottom = frustum_bottom
            camera.frustum_top = frustum_top

    def first_scene_camera_transform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("first_scene_camera_transform")
        camera = self._scene.camera[0]
        return camera.pos.copy(), camera.forward.copy(), camera.up.copy()

    def render(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("render", out=out, width=width, height=height)

        height = height or self._height
        width = width or self._width
        rect = mj.MjrRect(0, 0, width, height)

        original_flags = self._scene.flags.copy()

        # Enable shadow rendering (required for shadows to appear in rendered images)
        # Shadows are controlled by lights with castshadow enabled
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        # Using segmented rendering for depth makes the calculated depth more
        # accurate at far distances.
        if self._depth_rendering or self._segmentation_rendering:
            self._scene.flags[mj.mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mj.mjtRndFlag.mjRND_IDCOLOR] = True

        # Upload textures to GPU before rendering if textures have been modified
        # This is necessary when textures are modified via model.tex_data
        # Only upload when needed to avoid performance overhead
        if self._textures_need_upload:
            self.upload_textures()
            self._textures_need_upload = False

        if self._depth_rendering:
            out_shape = (rect.height, rect.width)
            out_dtype = np.float32
        else:
            out_shape = (rect.height, rect.width, 3)
            out_dtype = np.uint8

        if out is None:
            out = np.empty(out_shape, dtype=out_dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(
                    f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`"
                    " instead. When using depth rendering, the out array should be of"
                    " shape `(width, height)` and otherwise (width, height, 3)."
                    f" Got `(self.height, self.width)={(self.height, self.width)}` and"
                    f" `self._depth_rendering={self._depth_rendering}`."
                )

        # Render scene and read contents of RGB and depth buffers.
        mj.mjr_render(rect, self._scene, self._mjr_context)

        if self._depth_rendering:
            mj.mjr_readPixels(rgb=None, depth=out, viewport=rect, con=self._mjr_context)

            # Get the distances to the near and far clipping planes.
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent

            # Calculate OpenGL perspective matrix values in float32 precision
            # so they are close to what glFrustum returns
            # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
            zfar = np.float32(far)
            znear = np.float32(near)
            c_coef = -(zfar + znear) / (zfar - znear)
            d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

            # In reverse Z mode the perspective matrix is transformed by the following
            c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
            d_coef = np.float32(-0.5) * d_coef

            # We need 64 bits to convert Z from ndc to metric depth without noticeable
            # losses in precision
            out_64 = out.astype(np.float64)

            # Undo OpenGL projection
            # Note: We do not need to take action to convert from window coordinates
            # to normalized device coordinates because in reversed Z mode the mapping
            # is identity
            out_64 = d_coef / (out_64 + c_coef)

            # Cast result back to float32 for backwards compatibility
            # This has a small accuracy cost
            out[:] = out_64.astype(np.float32)

            # Reset scene flags.
            np.copyto(self._scene.flags, original_flags)
        elif self._segmentation_rendering:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

            # Convert 3-channel uint8 to 1-channel uint32.
            image3 = out.astype(np.uint32)
            segimage = image3[:, :, 0] + image3[:, :, 1] * (2**8) + image3[:, :, 2] * (2**16)
            # Remap segid to 3-channel (object ID, object type, body ID) triplet
            # Seg ID 0 is background -- will be remapped to (-1, -1, -1).

            # Find the maximum segment ID in the image to size the output array correctly
            max_segid = np.max(segimage) if segimage.size > 0 else 0

            # Create output array with size to accommodate all possible segment IDs
            # Add 1 to account for 0-based indexing and ensure we have enough space
            segid2output = np.full((max_segid + 1, 3), fill_value=-1, dtype=np.int32)

            visible_geoms = [g for g in self._scene.geoms[: self._scene.ngeom] if g.segid != -1]
            visible_segids = np.array([g.segid + 1 for g in visible_geoms], np.int32)
            visible_objid = np.array([g.objid for g in visible_geoms], np.int32)
            visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)
            visible_bodyid = np.array(
                [self.geomid_to_bodyid(g.objid) for g in visible_geoms], np.int32
            )

            # Only set values for valid segment IDs that are within bounds
            valid_mask = (visible_segids >= 0) & (visible_segids < segid2output.shape[0])
            if np.any(valid_mask):
                segid2output[visible_segids[valid_mask], 0] = visible_objid[valid_mask]
                segid2output[visible_segids[valid_mask], 1] = visible_objtype[valid_mask]
                segid2output[visible_segids[valid_mask], 2] = visible_bodyid[valid_mask]

            out = segid2output[segimage]

            # Reset scene flags.
            np.copyto(self._scene.flags, original_flags)
        else:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            if not _MS_RENDER_DIET:
                mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

        return out

    def render_rgb(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("render_rgb", out=out, width=width, height=height)

        height = height or self._height
        width = width or self._width
        rect = mj.MjrRect(0, 0, width, height)

        # Enable shadow rendering (required for shadows to appear in rendered images)
        # Shadows are controlled by lights with castshadow enabled
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        # Using segmented rendering for depth makes the calculated depth more
        # accurate at far distances.
        if self._depth_rendering or self._segmentation_rendering:
            self._scene.flags[mj.mjtRndFlag.mjRND_SEGMENT] = True
            self._scene.flags[mj.mjtRndFlag.mjRND_IDCOLOR] = True

        # Upload textures to GPU before rendering if textures have been modified
        # This is necessary when textures are modified via model.tex_data
        # Only upload when needed to avoid performance overhead
        if self._textures_need_upload:
            self.upload_textures()
            self._textures_need_upload = False

        if self._depth_rendering:
            out_shape = (rect.height, rect.width)
            out_dtype = np.float32
        else:
            out_shape = (rect.height, rect.width, 3)
            out_dtype = np.uint8

        if out is None:
            out = np.empty(out_shape, dtype=out_dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(
                    f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`"
                    " instead. When using depth rendering, the out array should be of"
                    " shape `(width, height)` and otherwise (width, height, 3)."
                    f" Got `(self.height, self.width)={(self.height, self.width)}` and"
                    f" `self._depth_rendering={self._depth_rendering}`."
                )

        # Render scene and read contents of RGB and depth buffers.
        mj.mjr_render(rect, self._scene, self._mjr_context)

        if self._depth_rendering:
            mj.mjr_readPixels(rgb=None, depth=out, viewport=rect, con=self._mjr_context)
        elif self._segmentation_rendering:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
        else:
            mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            if not _MS_RENDER_DIET:
                mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)

        return out

    def upload_textures(self, data: mj.MjData | None = None) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("upload_textures", data=data)
        if self.model.ntex == 0:
            log.debug("upload_textures(): Skipping - no textures in model (ntex == 0)")
            return

        for tex_id in range(self.model.ntex):
            mj.mjr_uploadTexture(self.model, self._mjr_context, tex_id)

    def mark_textures_dirty(self) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("mark_textures_dirty")
        self._textures_need_upload = True

    def update(
        self,
        data: mj.MjData,
        camera: int | str | mj.MjvCamera = -1,
        scene_option: mj.MjvOption | None = None,
    ) -> None:
        if getattr(self, "_actor_proxy", False):
            return self._actor.call("update", data, camera=camera, scene_option=scene_option)

        if not isinstance(camera, mj.MjvCamera):
            camera_id = camera
            if isinstance(camera_id, str):
                camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA.value, camera_id)
                if camera_id == -1:
                    raise ValueError(f'The camera "{camera}" does not exist.')
            if camera_id < -1 or camera_id >= self.model.ncam:
                raise ValueError(
                    f"The camera id {camera_id} is out of range [-1, {self.model.ncam})."
                )

            # Render camera.
            camera = mj.MjvCamera()
            camera.fixedcamid = camera_id

            # Defaults to mjCAMERA_FREE, otherwise mjCAMERA_FIXED refers to a
            # camera explicitly defined in the model_bindings.
            if camera_id == -1:
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultFreeCamera(self.model, camera)
            else:
                camera.type = mj.mjtCamera.mjCAMERA_FIXED

        scene_option = scene_option or self._scene_option
        mj.mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )

    def close(self) -> None:
        if getattr(self, "_actor_proxy", False):
            if getattr(self, "_closed", False):
                return
            self._closed = True
            self._actor.close(wait=not _actor_async_close_enabled())
            return

        if hasattr(self, "_mjr_context") and self._mjr_context:
            mjr_context = self._mjr_context
            self._mjr_context = None
            namespace = getattr(self, "_filament_lock_namespace", None)
            owner_thread_id = getattr(self, "_filament_context_thread_id", None)
            label = "MjFilamentRenderer.MjrContext.free"
            if not schedule_filament_context_free(
                mjr_context,
                namespace,
                label,
                owner_thread_id=owner_thread_id,
            ):
                with filament_context_free_lock(namespace, label) as lock_info:
                    free_t0 = time.monotonic()
                    mjr_context.free()
                    free_s = time.monotonic() - free_t0
                log.info(
                    "MS_FILAMENT_MJR_CONTEXT_FREE_TIMING gpu=%s slots=%s "
                    "wait_s=%.3f free_s=%.3f lock_hold_s=%.3f deferred=0",
                    lock_info.get("gpu"),
                    lock_info.get("slots"),
                    float(lock_info.get("waited_s") or 0.0),
                    free_s,
                    float(lock_info.get("hold_s") or 0.0),
                )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")

    args = parser.parse_args()

    if args.model == "":
        print("Must provide a model via --model option")
        exit(1)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Given model '{args.model}' is not a valid file")
        exit(1)

    model = mj.MjModel.from_xml_path(model_path.as_posix())
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    renderer = MjFilamentRenderer(model=model)
    renderer.update(data=data)

    image = renderer.render()
    pil_image = Image.fromarray(image)
    pil_image.save("test_render.png")
