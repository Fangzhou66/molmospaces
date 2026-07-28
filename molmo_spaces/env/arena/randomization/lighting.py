"""Per-episode lighting randomization.

RENDERER DEFECT THIS FILE WORKS AROUND (measured 2026-07-27, mujoco 3.7.1)
=========================================================================
``mjModel.light_active`` is **one-way** under MuJoCo's Filament renderer.
``mjv_updateScene`` honours it correctly -- clearing it drops the light from
``mjvScene`` and ``scn.nlight`` falls -- but the Filament backend inside the
shipped ``libmujoco.so.3.7.1`` only ever *updates* and *adds* lights; it never
removes one it has already created.  So the light that vanished from mjvScene
keeps shining, frozen at whatever colour/pose it last had.

Measured on the production path (``mujoco.MjrContext`` + ``mjRENDERER=filament``,
Mesa llvmpipe, 256x256, 3-light scene).  Filament's temporal dither + FXAA make
two *identical* renders differ by up to 90/255 on ~68% of pixels, so the honest
statistic is mean-absolute-difference, whose render-to-render floor is 0.558:

    mutation applied after the context exists   scn.nlight   meanAbsDiff
    ------------------------------------------  -----------  -----------
    (nothing -- renderer noise floor)              4 -> 4         0.558
    light_active[0] = 0                            4 -> 3         0.558  <- NO-OP
    light_diffuse[0] = 0                           4 -> 4        77.487  <- works
    light_diffuse/specular/ambient[0] = 0          4 -> 4        77.487  <- works
    light_d/s/a[0] = 0  AND  light_active[0] = 0   4 -> 3         0.558  <- NO-OP
    light_intensity[:] = 0    (and x20)            4 -> 4         0.558  <- NO-OP

The shape of the defect is "Filament bakes most of a light at creation time and
only refreshes a subset per frame".  ``light_intensity`` is the clean proof:
writing it *before* the context exists changes the frame (mean 89.83 -> 24.54,
identical to authoring the same value in the XML), writing the same value
*after* does nothing (mad 0.438 == noise).  It is not "ignored", it is
creation-time only -- exactly like ``light_active`` and ``light_castshadow``.
Per-frame, only pose (``light_pos``/``light_dir``, via mjData's xpos/xdir) and
``light_diffuse`` get through.

Read the last two rows before touching ``set_active``:

  * Zeroing the colour works, so "light off" is expressed as a **mute**: save
    diffuse/specular/ambient and write zeros.
  * Clearing ``light_active`` as well **re-breaks it**.  Once the light is out
    of mjvScene, the zeroed colour never ships, and Filament keeps the stale
    *bright* copy.  Keeping the model "self-consistent" by also writing
    ``light_active = 0`` is therefore exactly the wrong instinct -- it looks
    tidier and silently restores the original bug.  ``get_active()`` reports
    the logical state instead; ``model.light_active`` stays 1 for a muted
    light on purpose.

Equivalence of a muted light to a genuinely absent one was measured against a
ground-truth context built with the light already inactive (so Filament never
created it).  Muting reproduces it to within meanAbsDiff 0.032 / 0.58% of
pixels (max 90/255, localised).  The residual is the shadow the now-black light
still casts: ``light_castshadow`` is *also* ignored by this renderer (measured
0.000 diff with ``mjRND_SHADOW`` on), so the shadow cannot be switched off
client-side.  0.58% of pixels of residual is the price; the status quo was
0.00% -- a byte-identical frame, i.e. the randomizer's decision thrown away.

Other per-light fields measured dead on a live model, listed so nobody adds a
randomizer for them expecting pixels to move.  ``light_specular`` and
``light_ambient`` are ignored outright (0.000 diff driving them to both 0.0 and
1.0 with diffuse untouched); ``light_intensity`` and ``light_castshadow`` are
creation-time only.  That makes three of this class's six knobs no-ops on
Filament today -- ``randomize_specular``, ``randomize_ambient`` and (before
this workaround) ``randomize_active``.  Only ``randomize_position`` /
``randomize_direction`` / ``randomize_diffuse`` were ever reaching the frame.
We still write specular/ambient when muting: it costs nothing, keeps the model
honest for non-Filament consumers, and is correct the day a renderer reads them.

One more Filament limitation worth knowing before writing a lighting fixture or
authoring a scene: **only a single directional light is honoured**.  A scene
with two ``directional="true"`` lights renders as if the second were not there,
and zeroing that second light's diffuse moves the frame by the noise floor.
Point/spot lights are not subject to this.

Both defects are in the shipped ``libmujoco.so.3.7.1`` binary, which cannot be
rebuilt on this machine, hence a client-side workaround rather than a fix.
Scope: this restores per-episode light on/off on the **local** render arm only.
On the v2 render-service arm no appearance reaches the server at all after
``init_model(mjb)`` -- see the KNOWN GAP comment in
``molmo_spaces/renderer/filament_rendering.py`` -- so nothing here helps there
until the protocol grows an appearance channel.  When it does, muting survives
the move for free: ``light_diffuse`` is already part of the mjb content key
(``_SIG_APPEARANCE_FIELDS``), whereas an unchanged ``light_active`` is not a
signal the service could act on.
"""

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation

# The three per-light colour channels a mute has to zero.  Only ``diffuse``
# currently reaches the Filament frame (see module docstring); the other two
# are included so the model state matches the intent for any other consumer.
_LIGHT_COLOR_FIELDS = ("diffuse", "specular", "ambient")


class LightingRandomizer:
    """
    Randomizer for lighting properties in MuJoCo simulations.

    Based on the mujoco-py LightingModder implementation, adapted to work
    with MjModel and MjData directly (instead of MjSim).

    Args:
        model (MjModel): MuJoCo model
        random_state (np.random.RandomState | None): Random state for reproducibility.
            If None, uses global numpy random state.
        light_names (list[str] | None): List of light names to randomize.
            If None, randomizes all lights in the model.
        randomize_position (bool): If True, randomizes light position
        randomize_direction (bool): If True, randomizes light direction
        randomize_specular (bool): If True, randomizes specular color
        randomize_ambient (bool): If True, randomizes ambient color
        randomize_diffuse (bool): If True, randomizes diffuse color
        randomize_active (bool): If True, randomizes whether light is active
        position_perturbation_size (float): Magnitude of position randomization
        direction_perturbation_size (float): Magnitude of direction randomization in radians
        specular_perturbation_size (float): Magnitude of specular color randomization
        ambient_perturbation_size (float): Magnitude of ambient color randomization
        diffuse_perturbation_size (float): Magnitude of diffuse color randomization

    Note:
        MjData should be passed to the randomize() method, not to __init__.
    """

    def __init__(
        self,
        model: MjModel,
        random_state: np.random.RandomState | None = None,
        light_names: list[str] | None = None,
        randomize_position: bool = True,
        randomize_direction: bool = True,
        randomize_specular: bool = True,
        randomize_ambient: bool = True,
        randomize_diffuse: bool = True,
        randomize_active: bool = True,
        position_perturbation_size: float = 0.1,
        direction_perturbation_size: float = 0.35,  # ~20 degrees
        specular_perturbation_size: float = 0.1,
        ambient_perturbation_size: float = 0.1,
        diffuse_perturbation_size: float = 0.1,
    ):
        self.model = model

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        # Get light IDs from model (use IDs directly since lights may not have names)
        if light_names is None:
            # Use all light IDs (0 to nlight-1)
            self.light_ids = list(range(model.nlight))
        else:
            # Convert light names to IDs
            self.light_ids = []
            for name in light_names:
                light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
                if light_id >= 0:
                    self.light_ids.append(light_id)

        # Debug: print detected lights
        if model.nlight == 0:
            print(f"   Warning: No lights in model (model.nlight={model.nlight})")
        else:
            print(
                f"   Found {len(self.light_ids)} lights (model.nlight={model.nlight}): IDs {self.light_ids}"
            )

        self.randomize_position = randomize_position
        self.randomize_direction = randomize_direction
        self.randomize_specular = randomize_specular
        self.randomize_ambient = randomize_ambient
        self.randomize_diffuse = randomize_diffuse
        self.randomize_active = randomize_active

        self.position_perturbation_size = position_perturbation_size
        self.direction_perturbation_size = direction_perturbation_size
        self.specular_perturbation_size = specular_perturbation_size
        self.ambient_perturbation_size = ambient_perturbation_size
        self.diffuse_perturbation_size = diffuse_perturbation_size

        # Enable shadow casting for all lights by default.
        # NOTE: this is a no-op under the Filament renderer -- flipping
        # light_castshadow on a live model produced a 0.000 pixel diff even with
        # mjRND_SHADOW set (see module docstring).  Kept because the classic
        # OpenGL renderer in molmo_spaces/renderer/opengl_rendering.py does
        # honour it, and because it costs nothing.
        for light_id in self.light_ids:
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[light_id] = 1

        # Lights we have muted (see module docstring): light_id -> the
        # diffuse/specular/ambient the light should have if it were switched
        # back on.  A muted light keeps model.light_active == 1 and renders
        # black; this dict is the only place its real colour survives.
        self._muted: dict[int, dict[str, np.ndarray]] = {}

        self.save_defaults()

    def save_defaults(self):
        """
        Save default light parameter values from the current model state.
        """
        # Never bake a mute into the defaults: a muted light reads back as
        # all-zero colour, which would make "restore" mean "stay dark forever".
        self._unmute_all()
        self._defaults = {light_id: {} for light_id in self.light_ids}
        for light_id in self.light_ids:
            self._defaults[light_id]["pos"] = np.array(self.model.light_pos[light_id])
            self._defaults[light_id]["dir"] = np.array(self.model.light_dir[light_id])
            self._defaults[light_id]["specular"] = np.array(self.model.light_specular[light_id])
            self._defaults[light_id]["ambient"] = np.array(self.model.light_ambient[light_id])
            self._defaults[light_id]["diffuse"] = np.array(self.model.light_diffuse[light_id])
            self._defaults[light_id]["active"] = int(self.model.light_active[light_id])
            # Save castshadow if available (enables shadow casting)
            if hasattr(self.model, "light_castshadow"):
                self._defaults[light_id]["castshadow"] = int(self.model.light_castshadow[light_id])

    # ------------------------------------------------------------------
    # mute / unmute: the client-side stand-in for light_active (see module
    # docstring for the measurements that forced this shape).
    # ------------------------------------------------------------------
    def _mute(self, light_id: int):
        """Make a light contribute nothing, without removing it from mjvScene.

        Idempotent: muting an already-muted light keeps the *first* snapshot,
        so a double mute cannot record zeros as the light's real colour.
        """
        if light_id not in self._muted:
            self._muted[light_id] = {
                f: np.array(getattr(self.model, f"light_{f}")[light_id])
                for f in _LIGHT_COLOR_FIELDS
            }
        for f in _LIGHT_COLOR_FIELDS:
            getattr(self.model, f"light_{f}")[light_id] = 0.0

    def _unmute(self, light_id: int):
        """Put a muted light's colour back. No-op if it was never muted."""
        saved = self._muted.pop(light_id, None)
        if saved is None:
            return
        for f in _LIGHT_COLOR_FIELDS:
            getattr(self.model, f"light_{f}")[light_id] = saved[f]

    def _unmute_all(self):
        for light_id in list(self._muted):
            self._unmute(light_id)

    def is_muted(self, light_id: int) -> bool:
        """True if this light is switched off via the zero-colour workaround."""
        return light_id in self._muted

    def _get_color(self, field: str, light_id: int) -> np.ndarray:
        """Read one colour channel, honouring an active mute.

        Returns the *logical* colour (what the light would emit if switched on),
        so ``set_x(); get_x()`` round-trips even while muted.  Use
        ``get_active()`` / ``is_muted()`` to ask whether the light is on; use
        ``model.light_<field>`` directly if you want the raw renderer input.
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        if light_id in self._muted:
            return np.array(self._muted[light_id][field])
        return np.array(getattr(self.model, f"light_{field}")[light_id])

    def _set_color(self, field: str, light_id: int, value: np.ndarray):
        """Write one colour channel, honouring an active mute.

        While a light is muted its model colour must stay zero, so a colour
        written now is parked in the mute snapshot and applied when the light
        is switched back on.  Without this, ``set_active(i, 0)`` followed by
        ``set_diffuse(i, ...)`` would quietly re-light the light -- an ordering
        trap, given that ``randomize()`` happens to do it the other way round
        today and nothing stops a future edit from reordering those loops.
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        value = np.clip(value, 0.0, 1.0)
        if light_id in self._muted:
            self._muted[light_id][field] = np.array(value)
            return
        getattr(self.model, f"light_{field}")[light_id] = value

    def restore_defaults(self, data: MjData | None = None):
        """
        Restore saved default light parameter values.

        A light whose default is *inactive* but which the renderer has already
        instantiated comes back as a mute, not as ``light_active = 0``: clearing
        light_active would leave Filament showing the stale bright light.

        Args:
            data (MjData | None): MuJoCo data for the forward pass. Pass it if
                you intend to render afterwards. Light *pose* only reaches the
                renderer through the derived ``data.light_xpos``/``light_xdir``,
                so without a forward pass the restored positions are invisible
                and the next frame keeps the previous episode's lighting
                geometry -- measured as a 2.450 meanAbsDiff residual against a
                0.558 noise floor. ``randomize()`` takes the same argument for
                the same reason.
        """
        for light_id in self.light_ids:
            self.set_pos(light_id, self._defaults[light_id]["pos"])
            self.set_dir(light_id, self._defaults[light_id]["dir"])
            self.set_specular(light_id, self._defaults[light_id]["specular"])
            self.set_ambient(light_id, self._defaults[light_id]["ambient"])
            self.set_diffuse(light_id, self._defaults[light_id]["diffuse"])
            self.set_active(light_id, self._defaults[light_id]["active"])
            # Restore castshadow if it was saved
            if hasattr(self.model, "light_castshadow") and "castshadow" in self._defaults[light_id]:
                self.model.light_castshadow[light_id] = self._defaults[light_id]["castshadow"]

        # Forward pass to propagate changes (see the docstring: light pose only
        # reaches the renderer via mjData).
        if data is not None:
            mujoco.mj_forward(self.model, data)

    def randomize(self, data: MjData | None = None):
        """
        Randomize all enabled light properties.

        Args:
            data (MjData | None): MuJoCo data for forward pass. If None, forward pass is skipped.
        """
        # Start every episode from a fully un-muted model.  Two reasons:
        #  * an episode that crashed between _mute() and the next randomize()
        #    would otherwise leave that light black forever;
        #  * the per-field randomizers below derive from self._defaults, so
        #    without this the *snapshot* a later _mute() takes could be a stale
        #    colour rather than this episode's.
        # This is also what makes the workaround non-accumulating: nothing is
        # ever multiplied or decremented, each episode rewrites from defaults.
        self._unmute_all()

        # Which lights are available to be switched on this episode.  Read
        # *after* the un-mute above, so it means "lights this scene actually
        # has", not "lights the previous episode happened to leave on".  That
        # is deliberate: letting episode N-1's coin flips pick the fallback
        # light for episode N is a cross-episode dependency nobody asked for.
        # Uses get_active(), not model.light_active, because a muted light
        # reads back as active=1 by design.
        active_lights_before = [
            light_id for light_id in self.light_ids if self.get_active(light_id) > 0
        ]

        for light_id in self.light_ids:
            if self.randomize_position:
                self._randomize_position(light_id)

            if self.randomize_direction:
                self._randomize_direction(light_id)

            if self.randomize_specular:
                self._randomize_specular(light_id)

            if self.randomize_ambient:
                self._randomize_ambient(light_id)

            if self.randomize_diffuse:
                self._randomize_diffuse(light_id)

            if self.randomize_active:
                self._randomize_active(light_id)

            # Enable shadow casting for all lights.  No-op under Filament (see
            # __init__), kept for the OpenGL renderer.
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[light_id] = 1

        # Ensure at least one light is on, so the scene is not pitch black.
        # MUST go through get_active()/set_active(): model.light_active reads
        # back 1 for a muted light, so testing it directly here would conclude
        # "a light is still on" while every light in the scene is black, and
        # writing it directly would leave the fallback light muted-but-flagged-
        # active, i.e. still black.
        active_lights_after = [
            light_id for light_id in self.light_ids if self.get_active(light_id) > 0
        ]

        if len(active_lights_after) == 0 and len(self.light_ids) > 0:
            # All lights were turned off - re-enable at least one (prefer the first one that was active before)
            if active_lights_before:
                # Re-enable the first light that was active before
                self.set_active(active_lights_before[0], 1)
            else:
                # If no lights were active before, enable the first light
                self.set_active(self.light_ids[0], 1)

        # Forward pass to propagate changes
        if data is not None:
            mujoco.mj_forward(self.model, data)

    def _randomize_position(self, light_id: int):
        """
        Randomize position of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size,
            high=self.position_perturbation_size,
            size=3,
        )
        new_pos = self._defaults[light_id]["pos"] + delta_pos
        self.set_pos(light_id, new_pos)

    def _randomize_direction(self, light_id: int):
        """
        Randomize direction (orientation) of a specific light.

        Args:
            light_id (int): ID of the light
        """
        # Sample a random axis and angle for rotation
        random_axis = self.random_state.uniform(-1, 1, size=3)
        axis_norm = np.linalg.norm(random_axis)
        if axis_norm > 1e-6:
            random_axis = random_axis / axis_norm
        else:
            random_axis = np.array([0, 0, 1])  # Fallback to z-axis

        random_angle = self.random_state.uniform(
            -self.direction_perturbation_size, self.direction_perturbation_size
        )

        # Create rotation from axis-angle
        rotation = Rotation.from_rotvec(random_axis * random_angle)

        # Apply rotation to default direction
        default_dir = self._defaults[light_id]["dir"]
        default_dir_norm = np.linalg.norm(default_dir)
        if default_dir_norm > 1e-6:
            default_dir_normalized = default_dir / default_dir_norm
        else:
            default_dir_normalized = np.array([0, 0, -1])  # Default downward direction

        new_dir = rotation.apply(default_dir_normalized)
        # Normalize the new direction to ensure it's a unit vector
        new_dir_norm = np.linalg.norm(new_dir)
        if new_dir_norm > 1e-6:
            new_dir = new_dir / new_dir_norm
        else:
            new_dir = default_dir_normalized

        # Scale back to original magnitude if default had non-unit length
        new_dir = new_dir * default_dir_norm if default_dir_norm > 1e-6 else new_dir

        self.set_dir(light_id, new_dir)

    def _randomize_specular(self, light_id: int):
        """
        Randomize specular color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.specular_perturbation_size,
            high=self.specular_perturbation_size,
            size=3,
        )
        new_specular = np.clip(self._defaults[light_id]["specular"] + delta, 0.0, 1.0)
        self.set_specular(light_id, new_specular)

    def _randomize_ambient(self, light_id: int):
        """
        Randomize ambient color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.ambient_perturbation_size,
            high=self.ambient_perturbation_size,
            size=3,
        )
        new_ambient = np.clip(self._defaults[light_id]["ambient"] + delta, 0.0, 1.0)
        self.set_ambient(light_id, new_ambient)

    def _randomize_diffuse(self, light_id: int):
        """
        Randomize diffuse color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.diffuse_perturbation_size,
            high=self.diffuse_perturbation_size,
            size=3,
        )
        new_diffuse = np.clip(self._defaults[light_id]["diffuse"] + delta, 0.0, 1.0)
        self.set_diffuse(light_id, new_diffuse)

    def _randomize_active(self, light_id: int):
        """
        Randomize active state of a specific light.

        Independent coin flip per light, so P(at least one light off) is
        1 - 2**-nlight, minus the all-off case which randomize() undoes:
        0% at nlight=1 (the single light is always re-enabled), 75.3% at
        nlight=2, 87.6% at 3, 98.3% at 6 (measured over 20k draws).  Note the
        nlight=1 row -- for a one-light scene this knob has never done
        anything, workaround or not.

        Runs last in randomize()'s per-light loop on purpose: set_active(0)
        snapshots the colours the other randomizers just wrote.  set_diffuse()
        and friends are mute-aware so a reordering would not corrupt the
        snapshot, but this order keeps the intent obvious.

        Args:
            light_id (int): ID of the light
        """
        active = int(self.random_state.uniform() > 0.5)
        self.set_active(light_id, active)

    def get_pos(self, light_id: int) -> np.ndarray:
        """
        Get position of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (x, y, z) position
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_pos[light_id])

    def set_pos(self, light_id: int, value: np.ndarray):
        """
        Set position of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (x, y, z) position
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        self.model.light_pos[light_id] = value

    def get_dir(self, light_id: int) -> np.ndarray:
        """
        Get direction of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (x, y, z) direction vector
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_dir[light_id])

    def set_dir(self, light_id: int, value: np.ndarray):
        """
        Set direction of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (x, y, z) direction vector
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        # Normalize direction vector
        norm = np.linalg.norm(value)
        if norm > 0:
            value = value / norm
        self.model.light_dir[light_id] = value

    def get_active(self, light_id: int) -> int:
        """
        Get active state of a specific light.

        This is the *logical* state -- what the light is meant to be doing --
        which for a muted light is not the same as ``model.light_active``.
        Anything asking "is this light contributing to the image?" wants this,
        not the raw array.  See the module docstring.

        Args:
            light_id (int): ID of the light

        Returns:
            int: 1 if active, 0 if inactive
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        if light_id in self._muted:
            return 0
        return int(self.model.light_active[light_id])

    def set_active(self, light_id: int, value: int):
        """
        Set active state of a specific light.

        "Off" is implemented by zeroing the light's colour, NOT by clearing
        ``model.light_active``.  MuJoCo 3.7.1's Filament renderer never removes
        a light it has already created, so clearing light_active only hides the
        light from ``mjvScene`` -- Filament keeps drawing the last version it
        saw, at full brightness, and the frame comes back byte-identical.
        Measured: ``light_active[0] = 0`` -> meanAbsDiff 0.558 == the renderer's
        own noise floor; ``light_diffuse[0] = 0`` -> meanAbsDiff 77.487.

        Do not "tidy this up" by also writing ``light_active = 0`` for symmetry.
        That was measured too, and it puts the bug straight back: with the light
        out of mjvScene the zeroed colour never ships (meanAbsDiff 0.558 again).
        ``get_active()`` exists so callers can still read the intended state.

        The one case where clearing light_active *is* right is a light that is
        already inactive: the renderer has never created it, so leaving it
        absent is strictly better than instantiating a black one (no ghost
        shadow, no wasted light slot).  That case is handled below.

        Args:
            light_id (int): ID of the light
            value (int): 1 for active, 0 for inactive
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        if value:
            self._unmute(light_id)
            self.model.light_active[light_id] = 1
            return
        if int(self.model.light_active[light_id]) == 0:
            # Never handed to the renderer, so it was never created. Genuinely
            # absent beats black.
            self._muted.pop(light_id, None)
            return
        self._mute(light_id)

    def get_specular(self, light_id: int) -> np.ndarray:
        """
        Get specular color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) specular color
        """
        return self._get_color("specular", light_id)

    def set_specular(self, light_id: int, value: np.ndarray):
        """
        Set specular color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) specular color
        """
        self._set_color("specular", light_id, value)

    def get_ambient(self, light_id: int) -> np.ndarray:
        """
        Get ambient color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) ambient color
        """
        return self._get_color("ambient", light_id)

    def set_ambient(self, light_id: int, value: np.ndarray):
        """
        Set ambient color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) ambient color
        """
        self._set_color("ambient", light_id, value)

    def get_diffuse(self, light_id: int) -> np.ndarray:
        """
        Get diffuse color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) diffuse color
        """
        return self._get_color("diffuse", light_id)

    def set_diffuse(self, light_id: int, value: np.ndarray):
        """
        Set diffuse color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) diffuse color
        """
        self._set_color("diffuse", light_id, value)

    def update_model(self, model: MjModel):
        """
        Update the model reference.

        Args:
            model (MjModel): New MuJoCo model
        """
        # Drop mute bookkeeping *before* rebinding: those snapshots describe
        # the old model's lights, and save_defaults() un-mutes, which would
        # otherwise stamp the old scene's colours onto the new one.
        self._muted = {}
        self.model = model
        self.save_defaults()
