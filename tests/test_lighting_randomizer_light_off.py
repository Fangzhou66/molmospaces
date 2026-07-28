"""Gate: "this episode has the light off" must actually change the frame.

Background
==========
``mjModel.light_active`` is one-way under MuJoCo 3.7.1's Filament renderer.
``mjv_updateScene`` honours it (``scn.nlight`` drops), but the Filament backend
never removes a light it has already created, so switching an existing light
off mid-session returns a byte-identical frame.  ``LightingRandomizer`` used to
express "light off" exactly that way, which made the decision a silent no-op.
It now mutes the light instead -- zeroing diffuse/specular/ambient and
*deliberately leaving* ``light_active == 1``.  Full measurements and the
rejected alternatives are in the module docstring of
``molmo_spaces/env/arena/randomization/lighting.py``.

Two layers of gate, because they fail for different reasons:

  * ``TestLightOffIntent`` -- pure CPU, no renderer, always runs.  Pins the
    array-level contract: off means zeroed colour AND active still 1, on
    restores exactly, and repeated episodes do not drift.  This is what
    catches somebody "simplifying" set_active back to ``light_active = 0``.
  * ``TestLightOffReachesPixels`` -- renders.  Catches the case where the
    array-level contract is intact but the renderer stops honouring
    ``light_diffuse`` too (i.e. a mujoco upgrade moves the goalposts again).
    Skips cleanly when no usable Filament backend is present.

Why the render gate is CPU-only
-------------------------------
It forces Mesa's llvmpipe (software Vulkan) via ``VK_ICD_FILENAMES``.  That is
not a convenience, it is correctness: Filament's OpenGL backend on this
machine's H100 emits a constant black frame from its tone-mapping pass, and a
black frame does not change when you turn a light off -- a GL run would
"pass"/"fail" for reasons unrelated to lighting.  Filament's *hardware* Vulkan
backend wedges the GPU channel (Xid 109).  llvmpipe is the only backend here
that produces a correct image, and a 256x256 frame takes ~1s.

Why mean-absolute-difference and not byte equality
--------------------------------------------------
Filament's temporal dithering and FXAA are on by default, so two *identical*
renders differ by up to 90/255 on ~68% of pixels; only the mean is stable.
Measured on the 2-light fixture below: noise floor is ~0.5 meanAbsDiff, a light
going out is ~40-80.  The threshold sits two orders of magnitude clear of the
noise, so this does not flake.
"""

import os
import subprocess
import sys
import textwrap
import unittest

import mujoco as mj
import numpy as np

from molmo_spaces.env.arena.randomization.lighting import LightingRandomizer

# Two lights, so that switching one off leaves a scene that is still lit -- a
# fully black frame would be a degenerate way to "pass".  Bright floor plus a
# couple of geoms so the missing light has something to stop illuminating.
#
# The light the gate switches off (id 0, "key") is the DIRECTIONAL one on
# purpose.  Filament honours only a single directional light per scene: a
# fixture with two directional lights renders as if the second did not exist,
# and zeroing its diffuse moves the frame by 0.450 -- i.e. the noise floor --
# which would make this gate pass vacuously no matter what set_active() did.
# Measured on this fixture: switching the directional key off moves the frame
# by 45.6 meanAbsDiff against a 0.44 noise floor (a 100x margin); doing the
# same to the spot fill moves it by only 1.9.  Do not "simplify" the fixture by
# making both lights the same type.
_XML = """
<mujoco model="light_off_gate">
  <visual>
    <headlight ambient="0.05 0.05 0.05" diffuse="0.05 0.05 0.05"
               specular="0 0 0" active="1"/>
    <global offwidth="256" offheight="256"/>
  </visual>
  <worldbody>
    <light name="key"  pos="0.8 -0.8 1.6" dir="-0.4 0.4 -1" directional="true"
           diffuse="0.9 0.9 0.85" specular="0.4 0.4 0.4"/>
    <light name="fill" pos="-0.9 0.7 1.2" dir="0.5 -0.4 -1" directional="false"
           diffuse="0.5 0.55 0.7" specular="0.2 0.2 0.2"
           attenuation="1 0 0" range="8"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0"
          rgba="0.8 0.8 0.8 1"/>
    <geom name="b0" type="box" pos="-0.25 0 0.15" size="0.15 0.15 0.15"
          rgba="0.85 0.3 0.2 1"/>
    <geom name="b1" type="sphere" pos="0.3 0.1 0.16" size="0.16"
          rgba="0.3 0.5 0.9 1"/>
    <camera name="cam" pos="1.0 -1.2 0.9" xyaxes="0.77 0.64 0 -0.30 0.36 0.88"/>
  </worldbody>
</mujoco>
"""

_MESA_LVP_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

# meanAbsDiff over a uint8 RGB frame.  Anything above this is a real change;
# the renderer's own dither/FXAA noise measures ~0.5.
_NOISE_CEILING = 5.0

# The frame must not be blank, or "no change" would be vacuous.
_MIN_BASELINE_MEAN = 5.0


def _make_randomizer(model, **kwargs):
    """LightingRandomizer with every randomization off unless asked for.

    The gate drives set_active() directly; leaving pose/colour randomization on
    would move pixels for reasons unrelated to the light being off.
    """
    opts = dict(
        randomize_position=False,
        randomize_direction=False,
        randomize_specular=False,
        randomize_ambient=False,
        randomize_diffuse=False,
        randomize_active=False,
    )
    opts.update(kwargs)
    return LightingRandomizer(model=model, **opts)


class TestLightOffIntent(unittest.TestCase):
    """Array-level contract. No renderer, no GPU, always runs."""

    def setUp(self):
        self.model = mj.MjModel.from_xml_string(_XML)
        self.rnd = _make_randomizer(self.model)

    def test_off_zeroes_colour_and_keeps_light_active_set(self):
        # The whole point of the workaround. If this assertion is ever
        # "cleaned up" to expect light_active == 0, read the module docstring
        # of lighting.py first: it was measured, and it renders as a no-op.
        self.rnd.set_active(0, 0)
        np.testing.assert_array_equal(self.model.light_diffuse[0], np.zeros(3))
        np.testing.assert_array_equal(self.model.light_specular[0], np.zeros(3))
        np.testing.assert_array_equal(self.model.light_ambient[0], np.zeros(3))
        self.assertEqual(
            int(self.model.light_active[0]),
            1,
            "light_active must stay 1 for a muted light: clearing it removes the "
            "light from mjvScene, the zeroed colour never reaches Filament, and "
            "the frame comes back unchanged (the original bug).",
        )
        # ...while the randomizer still reports the light as off.
        self.assertEqual(self.rnd.get_active(0), 0)
        self.assertTrue(self.rnd.is_muted(0))
        # The other light is untouched.
        self.assertEqual(self.rnd.get_active(1), 1)
        self.assertGreater(float(self.model.light_diffuse[1].max()), 0.0)

    def test_on_restores_exactly(self):
        before = {
            f: np.array(getattr(self.model, f"light_{f}")[0])
            for f in ("diffuse", "specular", "ambient")
        }
        self.rnd.set_active(0, 0)
        self.rnd.set_active(0, 1)
        for f, v in before.items():
            np.testing.assert_array_equal(getattr(self.model, f"light_{f}")[0], v)
        self.assertEqual(int(self.model.light_active[0]), 1)
        self.assertEqual(self.rnd.get_active(0), 1)
        self.assertFalse(self.rnd.is_muted(0))

    def test_repeated_off_does_not_lose_the_colour(self):
        """A double mute must not snapshot the already-zeroed colour."""
        before = np.array(self.model.light_diffuse[0])
        for _ in range(5):
            self.rnd.set_active(0, 0)
        self.rnd.set_active(0, 1)
        np.testing.assert_array_equal(self.model.light_diffuse[0], before)

    def test_many_episodes_do_not_accumulate_drift(self):
        """200 randomize() calls must leave the lights recoverable, not decayed."""
        model = mj.MjModel.from_xml_string(_XML)
        rnd = _make_randomizer(
            model,
            randomize_position=True,
            randomize_direction=True,
            randomize_specular=True,
            randomize_ambient=True,
            randomize_diffuse=True,
            randomize_active=True,
        )
        data = mj.MjData(model)
        authored = {
            f: np.array(getattr(model, f"light_{f}").copy())
            for f in ("diffuse", "specular", "ambient", "pos", "dir")
        }
        saw_off = False
        for _ in range(200):
            rnd.randomize(data)
            saw_off = saw_off or any(rnd.get_active(i) == 0 for i in rnd.light_ids)
            # At least one light must always be lit.
            self.assertTrue(
                any(rnd.get_active(i) > 0 for i in rnd.light_ids),
                "randomize() must never leave every light off",
            )
            for i in rnd.light_ids:
                if rnd.get_active(i) > 0:
                    continue
                # A light reported off must actually be black in the model.
                np.testing.assert_array_equal(model.light_diffuse[i], np.zeros(3))
        self.assertTrue(saw_off, "fixture never exercised the light-off branch")

        rnd.restore_defaults()
        for f in ("diffuse", "specular", "pos", "dir"):
            np.testing.assert_allclose(
                getattr(model, f"light_{f}"),
                authored[f],
                atol=1e-12,
                err_msg=f"light_{f} drifted across 200 episodes",
            )
        for i in rnd.light_ids:
            self.assertEqual(rnd.get_active(i), 1)

    def test_crash_mid_episode_does_not_leave_a_light_dark(self):
        """An episode aborted while a light is muted must self-heal next time."""
        model = mj.MjModel.from_xml_string(_XML)
        rnd = _make_randomizer(model)
        authored = np.array(model.light_diffuse[0])
        rnd.set_active(0, 0)  # ... and then the episode blows up here.
        self.assertEqual(float(model.light_diffuse[0].max()), 0.0)

        # Next episode: randomize() un-mutes first, so nothing stays dark for
        # reasons the sampler never asked for.
        rnd.randomize(None)
        np.testing.assert_array_equal(model.light_diffuse[0], authored)
        self.assertEqual(rnd.get_active(0), 1)

    def test_colour_written_while_muted_is_applied_on_unmute(self):
        """set_active(0) then set_diffuse() must not silently re-light."""
        self.rnd.set_active(0, 0)
        self.rnd.set_diffuse(0, np.array([0.2, 0.4, 0.6]))
        np.testing.assert_array_equal(self.model.light_diffuse[0], np.zeros(3))
        np.testing.assert_allclose(self.rnd.get_diffuse(0), [0.2, 0.4, 0.6])
        self.rnd.set_active(0, 1)
        np.testing.assert_allclose(self.model.light_diffuse[0], [0.2, 0.4, 0.6])

    def test_light_inactive_at_load_stays_genuinely_absent(self):
        """A light the renderer never created is better left absent than black."""
        model = mj.MjModel.from_xml_string(_XML)
        model.light_active[0] = 0  # as if authored active="false"
        rnd = _make_randomizer(model)
        rnd.set_active(0, 0)
        self.assertEqual(int(model.light_active[0]), 0)
        self.assertFalse(rnd.is_muted(0))
        self.assertEqual(rnd.get_active(0), 0)
        # Turning it on still works (adding a light is the direction the
        # renderer handles correctly).
        rnd.set_active(0, 1)
        self.assertEqual(int(model.light_active[0]), 1)

    def test_save_defaults_never_captures_a_muted_colour(self):
        self.rnd.set_active(0, 0)
        self.rnd.save_defaults()
        self.assertGreater(float(self.model.light_diffuse[0].max()), 0.0)
        self.rnd.restore_defaults()
        self.assertGreater(float(self.model.light_diffuse[0].max()), 0.0)


# --------------------------------------------------------------------------
# Render gate.  Runs in a subprocess: creating a Filament context mutates
# process-global graphics state and cannot be undone, so it must not leak into
# the rest of the suite.
# --------------------------------------------------------------------------

_RENDER_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import mujoco as mj
    sys.path.insert(0, {repo!r})
    from molmo_spaces.env.arena.randomization.lighting import LightingRandomizer

    XML = {xml!r}
    W = H = 256

    m = mj.MjModel.from_xml_string(XML)
    d = mj.MjData(m); mj.mj_forward(m, d)
    ctx = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_150.value)
    scn = mj.MjvScene(m, maxgeom=200)
    cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_FIXED; cam.fixedcamid = 0
    opt = mj.MjvOption(); vp = mj.MjrRect(0, 0, W, H)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
    rgb = np.zeros((H, W, 3), np.uint8)

    def shot():
        # Filament pipelines the first frames; render a few and keep the last.
        for _ in range(4):
            mj.mjv_updateScene(m, d, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scn)
            mj.mjr_render(vp, scn, ctx)
            mj.mjr_readPixels(rgb, None, vp, ctx)
        return rgb.copy()

    def mad(a, b):
        return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

    base = shot()
    noise = mad(base, shot())

    rnd = LightingRandomizer(
        model=m, randomize_position=False, randomize_direction=False,
        randomize_specular=False, randomize_ambient=False,
        randomize_diffuse=False, randomize_active=False,
    )
    rnd.set_active(0, 0)
    off = shot()
    rnd.set_active(0, 1)
    back = shot()

    print("RESULT " + json.dumps(dict(
        baseline_mean=float(base.mean()),
        off_mean=float(off.mean()),
        noise=noise,
        off_delta=mad(base, off),
        restore_delta=mad(base, back),
        nlight=int(m.nlight),
    )))
    """
)


def _filament_probe():
    """Run the render probe in a subprocess. Returns (dict | None, reason)."""
    if not os.path.exists(_MESA_LVP_ICD):
        return None, f"Mesa software Vulkan ICD not found at {_MESA_LVP_ICD}"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = _MESA_LVP_ICD
    env["mjRENDERER"] = "filament"
    env.pop("DISPLAY", None)  # keep Filament off the GL backend (black frames)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RENDER_PROBE.format(repo=repo, xml=_XML)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, "filament render probe timed out"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout)[-600:]
        return None, f"filament render probe unavailable (rc={proc.returncode}): {tail}"
    import json

    return json.loads(line[len("RESULT ") :]), ""


class TestLightOffReachesPixels(unittest.TestCase):
    """The end-to-end gate: turning a light off must move the pixels."""

    @classmethod
    def setUpClass(cls):
        cls.result, cls.reason = _filament_probe()

    def setUp(self):
        if self.result is None:
            self.skipTest(self.reason)

    def test_baseline_frame_is_not_blank(self):
        # A black frame cannot show a lighting change, so a "pass" on a blank
        # baseline would be meaningless. This is the guard against the GL
        # backend's all-black tone-mapping output sneaking in.
        self.assertGreater(
            self.result["baseline_mean"],
            _MIN_BASELINE_MEAN,
            f"baseline frame is degenerate ({self.result}); no verdict possible",
        )

    def test_renderer_noise_is_below_the_threshold(self):
        self.assertLess(
            self.result["noise"],
            _NOISE_CEILING,
            f"renderer noise floor exceeds the gate threshold ({self.result}); "
            "the gate cannot distinguish signal from dither",
        )

    def test_switching_a_light_off_changes_the_frame(self):
        self.assertGreater(
            self.result["off_delta"],
            _NOISE_CEILING,
            "switching a light off produced no pixel change "
            f"({self.result}). Either set_active() regressed to writing "
            "model.light_active = 0 -- which MuJoCo 3.7.1's Filament renderer "
            "ignores for an already-created light -- or the renderer stopped "
            "honouring light_diffuse as well. See lighting.py's module "
            "docstring.",
        )
        self.assertLess(
            self.result["off_mean"],
            self.result["baseline_mean"],
            f"the frame should get darker, not brighter ({self.result})",
        )

    def test_switching_it_back_on_restores_the_frame(self):
        self.assertLess(
            self.result["restore_delta"],
            _NOISE_CEILING,
            f"set_active(.., 1) did not restore the original frame ({self.result}); "
            "episodes would accumulate lighting drift",
        )


if __name__ == "__main__":
    unittest.main()
