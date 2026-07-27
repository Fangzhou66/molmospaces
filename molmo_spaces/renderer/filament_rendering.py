import logging
import hashlib
import os
import time
from typing import Any

import mujoco as mj
import numpy as np

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer
from molmo_spaces.utils.filament_context_lock import (
    filament_context_creation_lock,
    filament_context_free_lock,
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
        **kwargs: Any,
    ) -> None:
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

        # Render-service routing (MS_RENDER_SERVICE_TAG/SLOT): plain-RGB
        # renders go to the per-GPU consolidated service (2026-07-05 audit:
        # cross-process Vulkan contention capped filament at ~16 renders/s/
        # GPU; the service is ~9-10x and pixmatch=1.000000). Depth/
        # segmentation and any failure fall back to the local context.
        self._service_slot = None
        self._service_last_data = None
        self._service_last_camera = None
        from molmo_spaces.renderer import render_service_client as _rsc

        if _rsc.service_enabled():
            try:
                import tempfile as _tf

                if os.environ.get(_rsc.TAGS_ENV):
                    self._service_slot = _rsc.claim_any_slot()
                else:
                    self._service_slot = _rsc.RenderServiceSlot()
                transport = os.environ.get("MS_RENDER_SERVICE_TRANSPORT", "").lower()
                if transport == "xmlfile":
                    if model_bindings is None or not model_bindings.xml_path:
                        raise RuntimeError(
                            "MS_RENDER_SERVICE_TRANSPORT=xmlfile requires a runtime XML path"
                        )
                    init_path = f"xmlfile:{model_bindings.xml_path}"
                else:
                    mjb_dir = os.environ.get(
                        "MS_RENDER_SERVICE_MJB_DIR", _tf.gettempdir()
                    )
                    # Content-keyed, write-once transport: benchmark tasks share
                    # compiled models heavily (canary 8757: 96/96 resets saved a
                    # byte-identical 528MB mjb; the synchronized rebuild wave was
                    # a 16GB NFS burst -> 300s+ resets -> mass engine quarantine).
                    # Same key => same file => the server can skip reloading too.
                    import hashlib as _hl

                    _h = _hl.blake2b(digest_size=8)
                    _h.update(np.int64(mj.mj_sizeModel(model)).tobytes())
                    for _arr in (
                        model.qpos0, model.body_pos, model.geom_pos,
                        model.geom_size, model.tex_adr, model.mesh_vertadr,
                    ):
                        _h.update(np.ascontiguousarray(_arr).tobytes())
                    mjb_path = os.path.join(
                        mjb_dir, f"rsvc_sig_{_h.hexdigest()}.mjb"
                    )
                    if not os.path.exists(mjb_path):
                        _tmp = f"{mjb_path}.tmp.{os.getpid()}"
                        mj.mj_saveModel(model, _tmp, None)
                        os.replace(_tmp, mjb_path)
                    init_path = (
                        f"mjbfile:{mjb_path}"
                        if transport == "mjbfile"
                        else mjb_path
                    )
                self._service_slot.init_model(init_path)
            except Exception:
                # FAIL LOUD by default (2026-07-26). Silently degrading to the
                # local renderer means a "service arm" can be a local arm, and
                # nobody can tell from the numbers -- exactly the shape of the
                # T4 gate defect, where a C++-only harness let a constant
                # world-origin camera pass for weeks. Any A/B, canary or
                # acceptance gate built on top of a silent fallback measures
                # the wrong thing.
                # MS_RENDER_SERVICE_ALLOW_FALLBACK=1 restores the old
                # degrade-and-continue behaviour for ad-hoc runs.
                if os.environ.get("MS_RENDER_SERVICE_ALLOW_FALLBACK") == "1":
                    log.exception(
                        "render service init failed; using local renderer "
                        "(MS_RENDER_SERVICE_ALLOW_FALLBACK=1)"
                    )
                    self._service_slot = None
                else:
                    raise

        # Protocol v1.3 gate. Deliberately OUTSIDE the try above: a stale
        # server has no cam[18] marker and would serialize an untouched
        # placeholder MjvCamera, rendering every frame from a constant
        # world-origin camera. That is the 2026-07-25 defect, and it is
        # invisible in throughput metrics -- so fail the process rather than
        # let another ladder be measured on the wrong pixels.
        if (
            self._service_slot is not None
            and not self._service_slot.glcam_capable
            and os.environ.get("MS_RENDER_SERVICE_GLCAM", "1") == "1"
        ):
            raise RuntimeError(
                "render_service_v2 predates protocol v1.3 (no cam[18] capability "
                "marker): it would render a constant world-origin camera. Rebuild "
                "the service, or set MS_RENDER_SERVICE_GLCAM=0 to deliberately "
                "reproduce the legacy (wrong-camera) arm."
            )

        # Turn off site rendering
        self._scene_option.sitegroup *= 0

        # Enable shadow rendering by default (shadows are controlled by lights with castshadow enabled)
        self._scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        if self._service_slot is not None:
            # Service owns the filament context for this slot; the engine
            # process touches no Vulkan at all (no per-engine context, no
            # arena pressure, no cross-process driver contention).
            self._mjr_context = None
            self._depth_rendering = False
            self._segmentation_rendering = False
            self._textures_need_upload = False
            log.info(
                "MS_RENDER_SERVICE active (slot %s): local MjrContext skipped",
                os.environ.get("MS_RENDER_SERVICE_SLOT"),
            )
            return

        _log_texture_cache_potential(model)
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
        return self._scene

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def enable_depth_rendering(self) -> None:
        self._segmentation_rendering = False
        self._depth_rendering = True

    def disable_depth_rendering(self) -> None:
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = True
        self._depth_rendering = False

    def disable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = False

    def geomid_to_bodyid(self, geomid):
        return self.model.geom_bodyid[geomid]

    def _service_cam(self, width: int, height: int):
        """Build the render-service camera payload -> ``(cam10, cam_tail)``.

        ``_render_frame`` writes the real pose onto ``scene.camera[*]``
        (``MjvGLCamera``: pos/forward/up) AFTER ``update()`` returns; the
        ``MjvCamera`` it hands to ``update()`` is a bare placeholder that has
        no such fields. So the resolved scene camera -- not
        ``_service_last_camera`` -- is the only object that carries the truth.
        """
        from molmo_spaces.renderer import render_service_client as _rsc

        if os.environ.get("MS_RENDER_SERVICE_GLCAM", "1") != "1":
            # Legacy arm, kept solely so the wrong-camera baseline is
            # reproducible for A/B.
            return _rsc.camera_to_cam10(
                self._service_last_camera, width, height
            ), None
        gl = self._scene.camera[0]
        fovy = float(self._model.vis.global_.fovy)
        return (
            _rsc.glcam_to_cam10(gl, fovy, width, height),
            _rsc.glcam_tail(gl, fovy),
        )

    def render(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        height = height or self._height
        width = width or self._width

        if (
            self._service_slot is not None
            and not self._depth_rendering
            and not self._segmentation_rendering
            and self._service_last_data is not None
        ):
            try:
                from molmo_spaces.renderer import render_service_client as _rsc

                ns = self._service_slot.nstate
                state = np.zeros(ns)
                mj.mj_getState(
                    self._model, self._service_last_data, state,
                    mj.mjtState.mjSTATE_INTEGRATION,
                )
                cam10, cam_tail = self._service_cam(width, height)
                blob = _rsc.derived_blob_from_data(
                    self._model, self._service_last_data
                )
                px = self._service_slot.render_rgb(
                    state, cam10, width, height, derived_blob=blob,
                    cam_tail=cam_tail,
                )
                if out is None:
                    return px.copy()
                out[...] = px
                return out
            except Exception:
                # The "fallback" below is not a fallback: under the service
                # self._mjr_context is None (set at __init__), so the local
                # path reaches mjr_readPixels(con=None) and raises TypeError,
                # killing the engine anyway -- reproduced in job 10608, where a
                # service-side geom error became a 60s TimeoutError and then a
                # TypeError. So the choice is between dying with a misleading
                # traceback and dying with the real one. Default to the real
                # one; MS_RENDER_SERVICE_ALLOW_FALLBACK=1 keeps the old path
                # for callers that genuinely hold a local context.
                if (os.environ.get("MS_RENDER_SERVICE_ALLOW_FALLBACK") != "1"
                        or self._mjr_context is None):
                    log.exception("render service call failed")
                    raise
                log.exception("render service call failed; falling back local")

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
        if self._mjr_context is not None:
            mj.mjr_render(rect, self._scene, self._mjr_context)
        elif self._depth_rendering:
            raise NotImplementedError(
                "depth rendering via the render service is not supported yet"
            )

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
            if self._mjr_context is not None:
                mj.mjr_readPixels(rgb=out, depth=None, viewport=rect, con=self._mjr_context)
            else:
                from molmo_spaces.renderer import render_service_client as _rsc

                ns = self._service_slot.nstate
                state = np.zeros(ns)
                mj.mj_getState(
                    self._model, self._service_last_data, state,
                    mj.mjtState.mjSTATE_INTEGRATION,
                )
                cam10, cam_tail = self._service_cam(width, height)
                blob = _rsc.derived_blob_from_data(self._model, self._service_last_data)
                out[...] = self._service_slot.render_rgb(
                    state, cam10, width, height, derived_blob=blob,
                    segmentation=True, cam_tail=cam_tail,
                )

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
        if self.model.ntex == 0:
            log.debug("upload_textures(): Skipping - no textures in model (ntex == 0)")
            return

        for tex_id in range(self.model.ntex):
            mj.mjr_uploadTexture(self.model, self._mjr_context, tex_id)

    def mark_textures_dirty(self) -> None:
        self._textures_need_upload = True

    def update(
        self,
        data: mj.MjData,
        camera: int | str | mj.MjvCamera = -1,
        scene_option: mj.MjvOption | None = None,
    ) -> None:
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

        self._service_last_data = data
        self._service_last_camera = camera

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
        if getattr(self, "_service_slot", None) is not None:
            # Release the render-service claim synchronously: a rebuild wave
            # where every engine holds its old slot until GC while claiming a
            # new one would exhaust the pool.
            self._service_slot.release()
            self._service_slot = None
        if hasattr(self, "_mjr_context") and self._mjr_context:
            with filament_context_free_lock(
                getattr(self, "_filament_lock_namespace", None),
                "MjFilamentRenderer.MjrContext.free",
            ) as lock_info:
                free_t0 = time.monotonic()
                self._mjr_context.free()
                free_s = time.monotonic() - free_t0
            log.info(
                "MS_FILAMENT_MJR_CONTEXT_FREE_TIMING gpu=%s slots=%s "
                "wait_s=%.3f free_s=%.3f lock_hold_s=%.3f",
                lock_info.get("gpu"),
                lock_info.get("slots"),
                float(lock_info.get("waited_s") or 0.0),
                free_s,
                float(lock_info.get("hold_s") or 0.0),
            )
        self._mjr_context = None


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
