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
  [app_off] appearance record            <-- protocol v1.4, see below
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

# The v1.2/v1.3 slot size. This is what the client REQUIRES a slot file to be,
# and it is deliberately NOT bumped by v1.4: the appearance region sits past
# the end of it, so a v1.4 client keeps working against a v1.2/v1.3 server
# (with the appearance channel reported as unavailable) instead of hanging in
# the "slot shm missing" size wait.
BASE_SLOT_BYTES = _PIX_OFF + _MAX_W * _MAX_H * 3
SLOT_BYTES = BASE_SLOT_BYTES

# ===========================================================================
# Protocol v1.4: the appearance channel  (defect B1, 2026-07-27)
# ===========================================================================
# THE DEFECT. Up to v1.3 the ONLY things that crossed the wire per frame were
# mjSTATE_INTEGRATION and the derived-kinematics blob. The model reached the
# service as a filesystem path, loaded once. So every per-episode appearance
# randomization -- geom_rgba (json_eval_task_sampler.py:725,931 -- the whole
# discriminative signal of col11_color), mat_rgba, the material scalars,
# tex_data, and every light parameter except pose -- was dropped on the floor,
# and scene reuse held the stale appearance for up to 64 episodes.
#
# WHY A CHANNEL AND NOT A RE-INIT. render_model_signature() already folds
# appearance into the mjb content key, so *changing the model* does reach the
# server -- via a new filename and a full reload. That is the right mechanism
# at SCENE granularity and the wrong one at EPISODE granularity: a reload is a
# mj_loadModel of a multi-hundred-MB mjb plus a SceneBridge rebuild (measured
# install_ms 3200 for a 15-geom toy; 30-70s for canary-8757 scenes), and the
# key is computed in MjFilamentRenderer.__init__, which runs once per env, not
# once per episode. Everything the randomizers do happens *after* that.
#
# WHAT TRAVELS, AND WHY THAT SPLIT. Sized against what the server actually
# reads (verified by reading the 3.10.0 filament compat layer, not by
# assumption):
#
#  TIER 1 -- small, per-frame, in-band. geom_rgba/geom_matid, mat_rgba and the
#    material scalars, and the light colour/attenuation set. These reach
#    filament through mjvGeom/mjvLight, which mjv_updateScene refills from
#    mjModel on EVERY frame (scene_geom_util.cc:157-183 reads geom.rgba /
#    geom.emission / geom.specular / geom.shininess / geom.reflectance and
#    model->mat_metallic/mat_roughness/mat_texid/mat_texrepeat/mat_texuniform;
#    scene_bridge.cc:210-215 reads scene_light.diffuse). So the server only has
#    to write them into its mjModel before mjv_updateScene -- no GPU work at
#    all. Cost is bounded by ngeom/nmat/nlight and is exactly
#    20*ngeom + 89*nmat + 58*nlight + framing -- MEASURED, not estimated: a
#    full record for rich_scene (15/6/3) is 1240B, and the same arithmetic
#    gives 38KB at 1000/200/8 and 134KB at 5000/400/16. Affordable every
#    frame, so it goes in the slot; and only CHANGED fields are sent, so the
#    steady-state record is empty and costs one memcmp per field.
#
#  TIER 2 -- large, per-episode, out-of-band. tex_data. The randomizer pool is
#    up to 200 textures of 512x512x3 (texture.py:105) = 150MB per episode worst
#    case, and whole-model tex_data has been measured at 192MB. A full
#    re-upload every episode is unaffordable in a 1MiB slot window and would be
#    24GB/s at 125 renders/s; even the tex_data *hash* is 214ms/192MB, so it
#    cannot be recomputed per frame either. Textures therefore travel (a) by
#    per-texture content digest, recomputed only when the caller says textures
#    are dirty, and (b) as a content-addressed sidecar file in the same shm
#    directory when the bytes do not fit inline. Content addressing means K
#    slots randomizing to the same bitmap pay for one file, and a repeat of a
#    previously seen bitmap is free.
#
#  NOT SENT BY DEFAULT -- light_intensity/range/bulbradius/type. These are read
#    ONCE, by LightManager's constructor (light_manager.cc:110-141), so the
#    server cannot honour them without rebuilding the SceneBridge (re-uploading
#    every mesh and texture). They are in the table, and the server does the
#    rebuild if you send them, but they are off the default set because nothing
#    in molmo_spaces randomizes them: LightingRandomizer writes pos/dir/
#    diffuse/specular/ambient/active only.
#
# WHERE IT LANDS, AND THE PRE-EXISTING HAZARD IT MUST NOT WORSEN. The derived
# blob is appended at _STATE_OFF + len(state) using the CLIENT's state length
# while the server reads it at kStateOff + nstate*8 using its OWN nstate, with
# no negotiated length and (before this change) no bounds check on either side.
# Because 200000 doubles are reserved before the pixel region, a disagreement
# over-reads into pixel memory instead of segfaulting -- silent corruption.
#
# The appearance region deliberately does NOT share that space. It is at a
# FIXED ABSOLUTE offset past the end of the pixel region:
#
#     _APP_OFF = BASE_SLOT_BYTES = 2260144
#
# so (a) its position does not depend on nstate and cannot be pushed around by
# a state-length disagreement, (b) an over-long appearance record runs off the
# end of the mapping (a fault) rather than into state or pixels, and (c) every
# pre-v1.4 offset is byte-identical, so the region is purely additive.
# Its length IS negotiated: the server advertises the capability in cam[19]
# (server-owned; the client packs cam[0..17] only, so it survives every
# render), the true capacity is the mapped file size minus _APP_OFF, and both
# sides bounds-check the record against it. render_rgb() additionally now
# asserts the state/derived-blob invariants that were missing -- see there.
#
# WIRING IT UP. This file provides the channel; two lines in
# molmo_spaces/renderer/filament_rendering.py switch it on, and until they
# land the channel is present but unfed (which the server reports as
# RSVC_V2_APPEARANCE records=0). That file is not owned by this change, so for
# the record the whole patch is:
#
#   1. MjFilamentRenderer.__init__, immediately after
#          self._service_slot.init_model(init_path)
#      add
#          self._service_slot.attach_model(model)
#      That is what makes geom_rgba / mat_* / light_* reach the pixels, and it
#      retires the KNOWN GAP block at filament_rendering.py:511-526.
#
#   2. MjFilamentRenderer.mark_textures_dirty(), on the service path, add
#          if self._service_slot is not None and self._service_slot.appearance:
#              self._service_slot.appearance.mark_textures_dirty()
#      Without it textures still travel, but only via
#      render_rgb(scan_textures=True), which re-hashes all of tex_data every
#      frame (214ms/192MB measured). Being told is the cheap path.
_APP_OFF = BASE_SLOT_BYTES
# What a v1.4 server reserves. Advisory only on this side: the authority is
# the mapped file size, so a server that reserves more or less still works.
_APP_BYTES = 1 << 20
SLOT_BYTES_V14 = _APP_OFF + _APP_BYTES
# Server -> client, stamped at slot-ready next to _GLCAM_CAP.
_APPEAR_CAP = 20260727.0
_APPEAR_CAP_OFF = _CAM_OFF + 19 * 8
# Client -> server, per render: cam[16] = appearance record length in bytes
# (0 = no record this frame). cam[16] was already being zeroed by every
# pre-v1.4 client, so an old client is indistinguishable from "no record".
_APP_LEN_CAM_IDX = 16

_APP_MAGIC = 0x5041534D  # 'MSAP' little-endian
_APP_VERSION = 1
_APP_HDR = struct.Struct("<IHHQIIQ")  # magic, ver, nfields, seq, payload, flags, rsv
_APP_FLD = struct.Struct("<HBBI")     # field_id, kind, pad, count
assert _APP_HDR.size == 32 and _APP_FLD.size == 8

_APP_FLAG_FULL = 1  # this record restates every managed field

# element kinds
_F32, _I32, _U8 = 0, 1, 2
_KIND_DTYPE = {_F32: "<f4", _I32: "<i4", _U8: "|u1"}
_KIND_SIZE = {_F32: 4, _I32: 4, _U8: 1}

# (field_id, mjModel attribute, element kind).
# FIELD IDS ARE FROZEN -- render_service_v2.cc switches on them and validates
# each count against its own model. Never renumber; only append.
APP_FIELDS_SURFACE = (
    (1, "geom_rgba", _F32),
    # geom_matid travels WITH geom_rgba and is not optional: the colour
    # randomizers write `geom_matid[g] = -1` immediately before
    # `geom_rgba[g] = c` (json_eval_task_sampler.py:724-725) precisely because
    # mjv_updateScene prefers mat_rgba over geom_rgba when matid >= 0. Shipping
    # geom_rgba without geom_matid would transport the colour and then let the
    # material override it -- a change that measures as "no pixel movement".
    (2, "geom_matid", _I32),
    (3, "mat_rgba", _F32),
    (4, "mat_emission", _F32),
    (5, "mat_specular", _F32),
    (6, "mat_shininess", _F32),
    (7, "mat_reflectance", _F32),
    (8, "mat_metallic", _F32),
    (9, "mat_roughness", _F32),
    (10, "mat_texid", _I32),
    (11, "mat_texrepeat", _F32),
    (12, "mat_texuniform", _U8),
    (25, "site_rgba", _F32),
    (26, "site_matid", _I32),
    (27, "tendon_rgba", _F32),
    (28, "skin_rgba", _F32),
)
APP_FIELDS_LIGHT = (
    (13, "light_diffuse", _F32),
    (14, "light_specular", _F32),
    (15, "light_ambient", _F32),
    (16, "light_active", _U8),
    (17, "light_castshadow", _U8),
    (18, "light_cutoff", _F32),
    (19, "light_exponent", _F32),
    (20, "light_attenuation", _F32),
)
# See the "NOT SENT BY DEFAULT" note above: latched by LightManager's ctor, so
# the server has to rebuild the SceneBridge to honour them.
APP_FIELDS_LIGHT_STRUCTURAL = (
    (21, "light_intensity", _F32),
    (22, "light_range", _F32),
    (23, "light_bulbradius", _F32),
    (24, "light_type", _I32),
)
APP_FIELDS_DEFAULT = APP_FIELDS_SURFACE + APP_FIELDS_LIGHT
APP_FIELDS_ALL = APP_FIELDS_DEFAULT + APP_FIELDS_LIGHT_STRUCTURAL

# Texture pseudo-fields. Both carry kind=_U8 and count=len(payload).
_TEX_INLINE = 64   # u32 tex_id, u32 nbytes, then nbytes of tex_data
_TEX_SIDECAR = 65  # ASCII manifest, one "tex_id nbytes filename" line per tex

APPEARANCE_ENV = "MS_RENDER_SERVICE_APPEARANCE"        # 1 (default) | 0
APPEARANCE_TEX_ENV = "MS_RENDER_SERVICE_APPEARANCE_TEX"  # 1 (default) | 0
# Bytes above which a changed texture goes to a sidecar file instead of inline.
# 128KiB keeps a whole 512x512x3 randomizer bitmap (768KB) out of the 1MiB
# window -- one of those inline would consume 75% of the record budget and
# starve the surface fields that share it.
_TEX_INLINE_MAX = int(os.environ.get("MS_RENDER_SERVICE_APPEARANCE_INLINE_MAX", 128 << 10))
# Per-process ceiling on sidecar bytes this client has created. /dev/shm is
# RAM: 64 engines x an unbounded texture cache is an OOM. When the budget is
# exceeded the oldest files THIS process created are unlinked (never another
# process's -- a sidecar is content-addressed and write-once, so someone else
# may still be reading it).
_TEX_SIDECAR_BUDGET = int(os.environ.get("MS_RENDER_SERVICE_APPEARANCE_TEX_BUDGET", 64 << 20))

TAG_ENV = "MS_RENDER_SERVICE_TAG"
SLOT_ENV = "MS_RENDER_SERVICE_SLOT"
TAGS_ENV = "MS_RENDER_SERVICE_TAGS"  # comma list: engines CLAIM a free slot
SHM_DIR_ENV = "MS_RENDER_SERVICE_SHM_DIR"  # tests only; production is /dev/shm

# Heartbeat staleness at which we log a WARNING. Deliberately NOT a death
# threshold -- see _service_liveness() for why nothing here is allowed to
# declare a service dead on a clock.
#
# Sizing (all numbers measured on this fleet):
#   - render_service_v2's dispatcher rewrites _heartbeat every 1.0s.
#   - Legitimate in-loop stalls: GpuInstall of one slot, and Engine teardown
#     measured at 10-13s, degrading to ~35s close_ms when concurrent frees
#     stampede the driver (see filament_context_lock._free_concurrency).
#   - The launcher's own outer bound is ALICE_SERVER_HEALTH_TIMEOUT_S=1200.
# 120s is ~100x the write period and ~3.5x the worst observed legitimate
# stall, so it never fires on a healthy-but-busy dispatcher, while staying far
# under 1200s so a genuinely wedged server is visible in logs long before the
# launcher's health timeout kills the run.
_HEARTBEAT_STALE_WARN_S = 120.0


def service_enabled() -> bool:
    if os.environ.get(TAGS_ENV):
        return True
    return bool(os.environ.get(TAG_ENV)) and os.environ.get(SLOT_ENV) is not None


def _shm_dir() -> str:
    return os.environ.get(SHM_DIR_ENV) or "/dev/shm"


def _slot_path(tag: str, slot: int) -> str:
    return os.path.join(_shm_dir(), f"msrender_{tag}_slot{slot}")


def _service_pid(tag: str) -> int | None:
    """The render service's own getpid(), as it stamped into its info.json.

    render_service_v2.WriteShardInfo() writes this BEFORE Dispatcher::
    CreateContext(), i.e. before the slow Vulkan bring-up, so the pid is
    readable for the entire startup window. It is written with a plain
    fopen("w")+fprintf, which is truncate-then-append and therefore NOT atomic:
    a reader can catch an empty or half-written file. Every parse failure
    returns None ("unknown"), never "dead" -- biasing an unreadable pid toward
    "assume alive" is what keeps a startup-race torn read from evicting a
    perfectly healthy service.
    """
    path = os.path.join(_shm_dir(), f"msrender_{tag}_info.json")
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
    except OSError:
        return None
    try:
        import json

        pid = int(json.loads(raw.decode("utf-8", "replace"))["pid"])
    except Exception:
        return None
    return pid if pid > 0 else None


def _pid_is_live(pid: int, tag: str) -> bool:
    """Kernel-authoritative liveness, with a PID-reuse guard.

    signal 0 is the actual liveness oracle; the /proc cmdline check exists only
    to defeat PID reuse (this fleet churns engine processes fast enough that a
    recycled pid is a real possibility over a multi-hour rollout). The service
    is reached via os.execvp() in render_service_pin_and_exec.py, so the pid is
    preserved across the exec and argv still carries the tag as argv[1].

    Note the asymmetry: we only ever return False on POSITIVE evidence of
    death. An unreadable /proc entry is treated as alive, because a false
    "dead" steals a slot out from under a running service -- strictly worse
    than the bug this function exists to fix.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another uid
    except OSError:
        return True
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [p for p in f.read().split(b"\0") if p]
    except OSError:
        return True  # no /proc (or raced with exit); trust signal 0
    if not argv:
        return True
    return any(tag.encode() == a or tag.encode() in a for a in argv)


def _heartbeat_age_s(tag: str) -> float | None:
    """Seconds since the dispatcher last touched the heartbeat, or None.

    None means "the file has never appeared", which is the NORMAL state for
    (a) a service still inside CreateContext() and (b) every MS_RENDER_STRICT_
    LSB=1 run, whose RunStrictLegacy path never calls Heartbeat() at all.
    """
    path = os.path.join(_shm_dir(), f"msrender_{tag}_heartbeat")
    try:
        return max(0.0, time.time() - os.stat(path).st_mtime)
    except OSError:
        return None


def _service_liveness(tag: str) -> tuple[str, str]:
    """Classify a service as live / starting / stalled / dead / unknown.

    WHY PID AND NOT AN MTIME THRESHOLD (2026-07-27, defect B2(b)).
    The bug was that claim_any_slot tested only os.path.exists() on the slot
    shm. /dev/shm is tmpfs: the segment outlives the process that created it,
    and the launcher only unlinks msrender_<tag>* on its CLEAN stop() path. A
    SIGKILL'd (or OOM-killed) service therefore leaves a full set of
    forever-claimable slot files behind, and the next engine claims a slot on a
    corpse -- the claim flock succeeds, because the dead owner's flock was
    already released by the kernel -- then hangs in init_model() until its
    180s flag-transition timeout.

    The obvious fix, "declare it dead if the heartbeat mtime is older than T",
    is unsafe HERE and no choice of T rescues it. render_service_v2's startup
    order is: MapSlot() all K slots (the shm files appear, full size, flag=
    kFree) -> WriteShardInfo() -> Dispatcher::CreateContext() -> Run(), and
    only Run() ever writes the heartbeat. Vulkan context creation is serialized
    at ~10s/engine, so a 32-engine spawn is >300s during which the slot files
    exist and the heartbeat file DOES NOT EXIST AT ALL. That is absence, not
    staleness: an mtime rule has nothing to compare against and would classify
    every legitimately-starting service as dead, which is the exact failure the
    launcher's ALICE_SERVER_HEALTH_TIMEOUT_S=1200 budget is meant to tolerate.

    flock would be the right primitive -- kernel-released on death, no
    threshold to tune, and it is what the claim files themselves already use.
    It is rejected only because the holder would have to be the server, and
    render_service_v2.cc contains no flock call anywhere; making it take one is
    a change to the ai2-mujoco tree, outside this file's ownership. See the
    report: that remains the recommended long-term fix.

    The pid gives us flock's decisive property -- the kernel reaps it on death,
    so there is NO threshold -- using a signal the server already publishes.
    Startup is covered for free: the pid is written before CreateContext(), so
    a 300s+ bring-up reads as "starting", indistinguishable in outcome from
    "live" and never from "dead".

    The heartbeat is retained, but demoted: it distinguishes "serving" from
    "still building its context", and flags a wedged dispatcher. It never
    condemns a service, because a hung-but-alive process still owns its Vulkan
    context and its slots -- taking them would double-bind the GPU.
    """
    pid = _service_pid(tag)
    if pid is None:
        # Pre-v2 render_service.cc writes neither info.json nor a heartbeat.
        # Refusing to claim would break every v1 deployment, so v1 keeps the
        # old existence-only semantics (and the old bug) rather than hard-fail.
        return "unknown", "no info.json (pre-v2 server, or startup race)"
    if not _pid_is_live(pid, tag):
        return "dead", f"server pid {pid} is gone; {_shm_dir()}/msrender_{tag}_* is stale"
    age = _heartbeat_age_s(tag)
    if age is None:
        return "starting", f"pid {pid} alive, no heartbeat yet (context bring-up)"
    if age > _HEARTBEAT_STALE_WARN_S:
        return "stalled", f"pid {pid} alive but heartbeat {age:.0f}s stale"
    return "live", f"pid {pid} alive, heartbeat {age:.1f}s old"


def claim_any_slot(nslots: int | None = None, wait_s: float = 300.0):
    """Claim a free slot on any advertised service via flock'd lockfiles.

    Zero-coordination allocation for group-wide env vars (alice engine
    processes share identical env). The claim is an flock held open for the
    claimant's lifetime: the kernel drops it on ANY process death, so engine
    crash/quarantine churn returns the slot as soon as the owner is gone
    (O_EXCL claim files leaked slots permanently — canary 8757 deadlocked
    on exactly that). Claim files are never unlinked while services run:
    unlink+recreate would let two claimants lock different inodes.

    B2(b) fix (2026-07-27): a slot is only a candidate if its SERVICE is not
    provably dead. File existence alone is not liveness -- /dev/shm is tmpfs
    and a SIGKILL'd service leaves its slot files behind forever. See
    _service_liveness() for the mechanism and for why pid, not heartbeat mtime,
    is the authority.
    """
    if nslots is None:
        nslots = int(os.environ.get("MS_RENDER_SERVICE_NSLOTS", "16"))
    tags = [t for t in os.environ[TAGS_ENV].split(",") if t]
    deadline = time.monotonic() + wait_s
    warned: set[str] = set()
    dead: dict[str, str] = {}
    while time.monotonic() < deadline:
        dead.clear()
        for tag in tags:
            state, detail = _service_liveness(tag)
            if state == "dead":
                # Do NOT reclaim. The file's standing convention (cf. the
                # _GLCAM_CAP check below and filament_context_lock's
                # "refusing to destroy without synchronization") is to fail
                # loud rather than proceed into a corrupt state, and automatic
                # reclamation here is exactly such a state: we cannot unlink
                # the claim file without breaking the single-inode invariant
                # above, and we cannot know that the corpse's GPU memory and
                # Vulkan queues have actually been reaped. Skip it; if every
                # advertised service is dead, raise below.
                dead[tag] = detail
                continue
            if state == "stalled" and tag not in warned:
                warned.add(tag)
                log.warning("render service %s: %s (claiming anyway)", tag, detail)
            for i in range(nslots):
                if not os.path.exists(_slot_path(tag, i)):
                    continue
                fd = os.open(_slot_path(tag, i) + ".claim", os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                # CHECK-THEN-CLAIM WINDOW. Ownership is decided solely by the
                # atomic LOCK_EX|LOCK_NB above on a stable inode, so two
                # claimants can never both win regardless of what the liveness
                # probe said -- the probe is a filter, never an arbiter. What
                # the window can still produce is a claim taken microseconds
                # before the service died, so we re-read liveness UNDER the
                # lock and drop the claim if it no longer holds. Closing the fd
                # releases the flock, so the slot stays available to whoever
                # can actually use it. (A service dying just after we return is
                # not closable by any check and is the caller's timeout to
                # handle.)
                state, detail = _service_liveness(tag)
                if state == "dead":
                    os.close(fd)
                    dead[tag] = detail
                    break
                os.ftruncate(fd, 0)
                os.write(fd, str(os.getpid()).encode())
                log.info("claimed render-service slot %s/%d (%s: %s)", tag, i, state, detail)
                s = RenderServiceSlot(tag=tag, slot=i)
                s._claim_fd = fd
                return s
        if dead and len(dead) == len(tags):
            # Every advertised service is a corpse. Spinning to wait_s would
            # bury the cause under a generic timeout; name the pids instead.
            raise RuntimeError(
                "all advertised render services are dead; refusing to claim a "
                "slot on a dead service: "
                + "; ".join(f"{t}: {d}" for t, d in sorted(dead.items()))
            )
        time.sleep(1.0)
    raise TimeoutError(
        "no free render-service slot"
        + (
            " (dead services skipped: "
            + "; ".join(f"{t}: {d}" for t, d in sorted(dead.items()))
            + ")"
            if dead
            else ""
        )
    )


class AppearanceChannel:
    """Turns "what changed on this mjModel since last frame" into a v1.4 record.

    Owns the dirty tracking. Two different mechanisms, because the two tiers
    have costs that differ by three orders of magnitude:

      * surface/light fields -- a stored copy per field and a bytes ``!=``
        (memcmp) each frame. Exact, no collision risk, and cheaper than
        hashing: ~100KB of memcmp at ngeom=5000. A hash would be the wrong
        tool here; the arrays are small enough that comparing them IS the
        cheap operation.

      * tex_data -- a per-texture blake2b digest, recomputed ONLY when
        :meth:`mark_textures_dirty` has been called. Hashing 192MB is 214ms;
        doing it per frame at 125 fps is 27x the entire frame budget. The
        caller has the information (task_sampler already calls
        ``renderer.mark_textures_dirty()`` after texture randomization) so the
        cheap correct thing is to be told, not to poll. ``scan_textures=True``
        forces the poll for callers that cannot be plumbed.

    "Dirty" is measured against the last record we SENT, not against the mjb,
    so the first record after :meth:`reset` is a FULL restatement of every
    managed field. That is what resynchronises us with a server whose model
    may carry another episode's deltas (the loader's content-key fast path
    reuses a live mjModel when the mjb path is unchanged).
    """

    def __init__(self, model, tag: str, fields=APP_FIELDS_DEFAULT,
                 textures: bool | None = None, shm_dir: str | None = None):
        self.model = model
        self.tag = tag
        self.shm_dir = shm_dir or _shm_dir()
        if textures is None:
            textures = os.environ.get(APPEARANCE_TEX_ENV, "1") == "1"
        self.textures_enabled = bool(textures)
        # Drop fields this mujoco build does not have rather than crashing:
        # the server validates counts anyway, and a renamed field should
        # degrade to "not transported" with a warning, not to an exception in
        # the render loop.
        self.fields = []
        for fid, name, kind in fields:
            arr = getattr(model, name, None)
            if arr is None:
                log.warning(
                    "appearance channel: mjModel has no field %r; it will not "
                    "be transported to the render service", name)
                continue
            self.fields.append((fid, name, kind))
        self._last: dict[str, bytes] = {}
        self._tex_last: list[bytes] | None = None
        self._tex_dirty = False
        self._sidecars: list[tuple[str, int]] = []  # (path, nbytes), oldest first
        self._sidecar_bytes = 0
        self.seq = 0
        # Snapshot the texture digests NOW so the first record does not
        # re-transport textures the server already has. This costs one full
        # tex_data hash (measured 214ms for 192MB) once per renderer
        # construction -- next to a 30-70s cold scene load and the mjb content
        # key's own hash of the same bytes. The alternative, "assume the
        # server's textures equal the mjb's", is true in production but a
        # landmine for anyone who mutates tex_data between mj_saveModel and
        # attach_model; paying 214ms once to not have that assumption is the
        # right trade.
        if self.textures_enabled:
            self._changed_textures()

    # -- dirty signalling ---------------------------------------------------
    def reset(self, resend_textures: bool = False) -> None:
        """Forget what the server knows: the next record is FULL.

        Textures are RE-SNAPSHOTTED rather than re-sent. After an init the
        server's model is the mjb, and the mjb is this model, so the correct
        baseline is "whatever tex_data holds right now" -- taking it costs one
        hash and transports nothing, where re-sending would be a
        multi-hundred-MB no-op. (The one way the server could still hold
        texture deltas across an init is the loader's content-key fast path;
        render_service_v2 refuses that path for any model it has
        texture-patched, Slot::tex_patched, rather than making every client
        pay here.) ``resend_textures=True`` forces the transport for a caller
        that knows the two have diverged.
        """
        self._last.clear()
        self._tex_last = None
        self._tex_dirty = bool(resend_textures)
        if self.textures_enabled and not resend_textures:
            self._changed_textures()  # re-baseline, send nothing

    def mark_textures_dirty(self) -> None:
        """Mirror of ``MjFilamentRenderer.mark_textures_dirty()``.

        On the local path that method flips ``_textures_need_upload`` so the
        next ``render()`` calls ``mjr_uploadTexture``. Under the service the
        client holds no ``mjrContext`` at all, so the equivalent is to let the
        next record carry the changed bytes and have the server call
        ``SceneBridge::UploadTexture``.
        """
        self._tex_dirty = True

    # -- record construction ------------------------------------------------
    def _array_bytes(self, name: str, kind: int) -> bytes:
        arr = np.ascontiguousarray(getattr(self.model, name))
        # mat_texuniform / light_active / light_castshadow come out of the
        # bindings as numpy bool, which is 1 byte but not uint8; astype makes
        # the wire type unambiguous instead of relying on that coincidence.
        return arr.astype(_KIND_DTYPE[kind], copy=False).tobytes()

    def _changed_textures(self) -> list[int]:
        """Texture ids whose bytes differ from the last record we sent."""
        import hashlib

        m = self.model
        ntex = int(m.ntex)
        if ntex == 0:
            return []
        data = np.asarray(m.tex_data, dtype=np.uint8)
        digests = []
        for t in range(ntex):
            adr = int(m.tex_adr[t])
            n = int(m.tex_width[t]) * int(m.tex_height[t]) * int(m.tex_nchannel[t])
            digests.append(
                hashlib.blake2b(data[adr:adr + n], digest_size=16).digest())
        if self._tex_last is None:
            changed = list(range(ntex))
        elif len(self._tex_last) != ntex:
            changed = list(range(ntex))  # model swapped under us
        else:
            changed = [t for t in range(ntex) if digests[t] != self._tex_last[t]]
        self._tex_last = digests
        return changed

    def _write_sidecar(self, digest_hex: str, payload: memoryview) -> str:
        """Content-addressed, write-once. Returns the bare filename."""
        name = f"msrender_{self.tag}_tex_{digest_hex}.bin"
        path = os.path.join(self.shm_dir, name)
        if not os.path.exists(path):
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, path)
            self._sidecars.append((path, len(payload)))
            self._sidecar_bytes += len(payload)
            while self._sidecar_bytes > _TEX_SIDECAR_BUDGET and len(self._sidecars) > 1:
                old, n = self._sidecars.pop(0)
                try:
                    os.unlink(old)
                except OSError:
                    pass
                self._sidecar_bytes -= n
        return name

    def build(self, capacity: int, scan_textures: bool = False) -> bytes | None:
        """Return the record for this frame, or None if nothing changed.

        ``capacity`` is the slot's real appearance-region size. Overflowing it
        RAISES: silently truncating would put us back where we started, with
        appearance quietly not reaching the pixels.
        """
        parts: list[bytes] = []
        nfields = 0
        full = not self._last

        for fid, name, kind in self.fields:
            raw = self._array_bytes(name, kind)
            if not raw:
                continue  # ngeom/nmat/nlight == 0
            if self._last.get(name) == raw:
                continue
            self._last[name] = raw
            count = len(raw) // _KIND_SIZE[kind]
            parts.append(_APP_FLD.pack(fid, kind, 0, count))
            parts.append(raw)
            parts.append(b"\0" * (-len(raw) % 8))
            nfields += 1

        if self.textures_enabled and (self._tex_dirty or scan_textures):
            self._tex_dirty = False
            manifest: list[str] = []
            m = self.model
            data = np.asarray(m.tex_data, dtype=np.uint8)
            for t in self._changed_textures():
                adr = int(m.tex_adr[t])
                n = int(m.tex_width[t]) * int(m.tex_height[t]) * int(m.tex_nchannel[t])
                blob = data[adr:adr + n]
                if n <= _TEX_INLINE_MAX:
                    body = struct.pack("<II", t, n) + blob.tobytes()
                    parts.append(_APP_FLD.pack(_TEX_INLINE, _U8, 0, len(body)))
                    parts.append(body)
                    parts.append(b"\0" * (-len(body) % 8))
                    nfields += 1
                else:
                    mv = memoryview(blob)
                    fname = self._write_sidecar(self._tex_last[t].hex(), mv)
                    manifest.append(f"{t} {n} {fname}")
            if manifest:
                body = ("\n".join(manifest)).encode()
                parts.append(_APP_FLD.pack(_TEX_SIDECAR, _U8, 0, len(body)))
                parts.append(body)
                parts.append(b"\0" * (-len(body) % 8))
                nfields += 1

        if nfields == 0:
            return None
        payload = b"".join(parts)
        self.seq += 1
        rec = _APP_HDR.pack(_APP_MAGIC, _APP_VERSION, nfields, self.seq,
                            len(payload), _APP_FLAG_FULL if full else 0, 0) + payload
        if len(rec) > capacity:
            raise ValueError(
                f"appearance record is {len(rec)}B but the slot reserves only "
                f"{capacity}B. Refusing to truncate -- a partial record is a "
                f"silently wrong image. Raise the server's kAppBytes, or lower "
                f"MS_RENDER_SERVICE_APPEARANCE_INLINE_MAX so large textures go "
                f"to sidecar files."
            )
        return rec


class RenderServiceSlot:
    def __init__(self, tag: str | None = None, slot: int | None = None):
        tag = tag or os.environ[TAG_ENV]
        slot = int(os.environ[SLOT_ENV]) if slot is None else slot
        self.slot = slot
        self.tag = tag  # which service this slot belongs to (claim diagnostics)
        path = _slot_path(tag, slot)
        deadline = time.monotonic() + 120
        while not (os.path.exists(path) and os.path.getsize(path) >= SLOT_BYTES):
            if time.monotonic() > deadline:
                raise TimeoutError(f"render service slot shm missing: {path}")
            time.sleep(0.1)
        self._f = open(path, "r+b")
        # Map whatever the server actually made, not what this client's
        # constants say it should be. A v1.4 server appends an appearance
        # region past BASE_SLOT_BYTES; mapping the true size is what lets a
        # v1.4 client talk to a v1.2 server (smaller file, no region) and a
        # v1.2 client talk to a v1.4 server (bigger file, region ignored)
        # without either one hard-coding the other's size.
        self._mapped = max(os.path.getsize(path), SLOT_BYTES)
        self._m = mmap.mmap(self._f.fileno(), self._mapped)
        self.nstate: int | None = None
        # Set from the server's cam[18] marker in init_model(); False until then.
        self.glcam_capable: bool = False
        # v1.4: set from cam[19]. app_capacity is the NEGOTIATED length of the
        # appearance region -- the server's advertisement bounded by the file
        # it actually created.
        self.appearance_capable: bool = False
        self.app_capacity: int = 0
        self.appearance: AppearanceChannel | None = None
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

    def attach_model(self, model, fields=APP_FIELDS_DEFAULT, textures=None) -> None:
        """Route this mjModel's appearance to the server on every render.

        Call it once, after :meth:`init_model`. From then on ``render_rgb``
        diffs the model's appearance arrays against what it last sent and
        ships only the difference -- so a randomizer that writes
        ``model.geom_rgba`` after the renderer was constructed reaches the
        pixels, which before protocol v1.4 it did not.

        Raises if the server predates v1.4, for the same reason the v1.3
        glcam gate raises: the alternative is an arm that silently renders
        stale colours and is indistinguishable, in every metric anyone
        collects, from an arm that works. ``MS_RENDER_SERVICE_APPEARANCE=0``
        deliberately reproduces the pre-v1.4 (frozen-appearance) behaviour.
        """
        if os.environ.get(APPEARANCE_ENV, "1") != "1":
            log.warning(
                "%s=0: appearance is NOT transported to the render service; "
                "per-episode geom_rgba/material/texture/light randomization "
                "will not reach the pixels", APPEARANCE_ENV)
            self.appearance = None
            return
        if not self.appearance_capable:
            raise RuntimeError(
                "render_service_v2 predates protocol v1.4 (no cam[19] "
                "appearance-capability marker): per-episode geom_rgba / "
                "mat_rgba / tex_data / light-parameter randomization would be "
                "silently dropped and every episode would render the mjb's "
                "appearance. Rebuild the service, or set "
                f"{APPEARANCE_ENV}=0 to accept the frozen-appearance arm."
            )
        self.appearance = AppearanceChannel(
            model, self.tag, fields=fields, textures=textures)

    def init_model(self, model_path: str, timeout_s: float = 180.0, model=None) -> int:
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
        self.appearance_capable = (
            self._mapped > _APP_OFF
            and struct.unpack_from("<d", self._m, _APPEAR_CAP_OFF)[0] == _APPEAR_CAP
        )
        # NEGOTIATED length: the region is whatever the server actually
        # allocated past the pixels, never a client-side constant.
        self.app_capacity = max(0, self._mapped - _APP_OFF) if self.appearance_capable else 0
        # An init resets the server's model to the mjb (or hands us a cached
        # one that may carry ANOTHER episode's deltas, via the loader's
        # content-key fast path). Either way what the server has is no longer
        # what we think we sent, so the next record must be a full restatement.
        if self.appearance is not None:
            self.appearance.reset()
        elif model is not None:
            self.attach_model(model)
        log.info(
            "render service slot %d ready (nstate=%d glcam=%s appearance=%s cap=%dB)",
            self.slot, self.nstate, self.glcam_capable,
            self.appearance_capable, self.app_capacity,
        )
        return self.nstate

    def render_rgb(self, state: np.ndarray, cam10, width: int, height: int,
                   derived_blob: bytes | None = None, segmentation: bool = False,
                   timeout_s: float = 60.0, cam_tail=None,
                   scan_textures: bool = False) -> np.ndarray:
        if width * height * 3 > _MAX_W * _MAX_H * 3:
            raise ValueError(f"resolution {width}x{height} exceeds protocol buffer")
        sb = state.tobytes()
        # THE MISSING ASSERTION (2026-07-27, shipped with v1.4).
        #
        # The derived blob is written at _STATE_OFF + len(sb) using OUR state
        # length, and read back at kStateOff + nstate*8 using the SERVER's
        # nstate. Nothing ever checked that those agree. When they do not, the
        # server's `take()` walks off the end of the blob -- and because
        # kMaxState=200000 doubles are reserved before the pixel region, the
        # over-read lands inside pixel memory instead of faulting. The result
        # is geometry silently placed from image bytes: an image that is wrong
        # in a way no "is the frame non-degenerate?" check can catch.
        #
        # nstate comes from the server (PublishDims writes it at base+4 and
        # init_model reads it back), so a mismatch means the caller built the
        # state array from something other than this slot's model. That is a
        # caller bug and it is not recoverable here -- refuse.
        if self.nstate is not None and len(sb) != self.nstate * 8:
            raise ValueError(
                f"state is {len(sb)}B ({len(sb) // 8} doubles) but the server "
                f"reports nstate={self.nstate}. The server would read the "
                f"derived blob from the wrong offset and over-read into pixel "
                f"memory. Rebuild the state with "
                f"mj_getState(model, data, np.zeros(slot.nstate), mjSTATE_INTEGRATION)."
            )
        blob_end = _STATE_OFF + len(sb) + (len(derived_blob) if derived_blob else 0)
        if blob_end > _PIX_OFF:
            raise ValueError(
                f"state+derived blob is {blob_end - _STATE_OFF}B, past the "
                f"{_PIX_OFF - _STATE_OFF}B reserved before the pixel region"
            )

        # v1.4 appearance record. Built BEFORE the flag store so the release
        # store below publishes it together with the state -- same fence, same
        # ordering guarantee the state transfer has always relied on.
        app_len = 0
        if self.appearance is not None:
            rec = self.appearance.build(self.app_capacity, scan_textures=scan_textures)
            if rec:
                self._m[_APP_OFF:_APP_OFF + len(rec)] = rec
                app_len = len(rec)

        # cam[0..17]; cam[18..19] belong to the server (capability markers).
        # cam[16] is the appearance record length -- 0 for every pre-v1.4
        # client, which is exactly "no record", so old clients need no change.
        cam18 = (tuple(cam10)
                 + (1.0 if derived_blob else 0.0, 1.0 if segmentation else 0.0)
                 + tuple(cam_tail if cam_tail is not None else (0.0,) * 4)
                 + (float(app_len), 0.0))
        struct.pack_into("<18d", self._m, _CAM_OFF, *cam18)
        self._m[_STATE_OFF:_STATE_OFF + len(sb)] = sb
        if derived_blob:
            off = _STATE_OFF + len(sb)
            self._m[off:off + len(derived_blob)] = derived_blob
        self._set_flag(2)
        self._wait((3, 4), timeout_s)
        if self._flag() == 4:
            # The server rejects a request it cannot serve correctly (bad
            # nstate, a blob that does not fit, an appearance record whose
            # field counts disagree with its model). Surface that instead of
            # returning whatever happens to be in the pixel region.
            raise RuntimeError(
                f"render service slot {self.slot} returned error for this "
                f"request; see the server's stderr (RSVC_V2_REJECT)"
            )
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
