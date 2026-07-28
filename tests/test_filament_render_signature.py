"""Regression tests for the render-service mjb content key (B1).

The mjb written by ``MjFilamentRenderer.__init__`` is a write-once,
content-addressed file in a shared directory, and under the render service it
is the *only* channel carrying model appearance to the server (per frame we
ship state + derived kinematics, nothing else). The original key hashed
geometry alone, so two episodes differing only in colour/texture/lighting
produced the same filename, the stale mjb was reused, and the rendered
observation kept the previous episode's appearance.

Every test here is pure CPU: it compiles small models and hashes arrays. No
context creation, no GPU, no rendering.
"""

import hashlib
import subprocess
import sys
import textwrap
import unittest

import mujoco as mj
import numpy as np

from molmo_spaces.renderer.filament_rendering import (
    _SIG_APPEARANCE_FIELDS,
    _SIG_ARRAY_FIELDS,
    render_model_signature,
)

# Two textures, two materials, two lights, two geoms -- enough for every
# appearance category to be independently perturbable. Textures are kept small
# so the suite stays fast; the hash path does not care about size.
_XML = """
<mujoco model="sig_test">
  <custom>
    <numeric name="filament_env_light_intensity" data="30000"/>
  </custom>
  <asset>
    <texture name="tex0" type="2d" builtin="checker" width="64" height="64"
             rgb1="1 0 0" rgb2="0 1 0"/>
    <texture name="tex1" type="2d" builtin="flat" width="64" height="64"
             rgb1="0 0 1"/>
    <material name="mat0" texture="tex0" rgba="0.5 0.4 0.3 1" specular="0.3"
              shininess="0.2"/>
    <material name="mat1" texture="tex1" rgba="0.1 0.2 0.9 1"/>
  </asset>
  <worldbody>
    <light name="light0" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"
           specular="0.1 0.1 0.1" ambient="0.05 0.05 0.05" castshadow="true"/>
    <light name="light1" pos="1 1 2" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>
    <body name="b0" pos="0 0 1">
      <freejoint/>
      <geom name="g0" type="box" size=".1 .1 .1" material="mat0"
            rgba="1 1 1 1"/>
    </body>
    <body name="b1" pos="1 0 1">
      <freejoint/>
      <geom name="g1" type="sphere" size=".1" material="mat1"
            rgba="0.2 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _legacy_signature(model: mj.MjModel) -> str:
    """The pre-fix key, verbatim, so the tests can prove the regression rather
    than merely assert the new behaviour."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.int64(mj.mj_sizeModel(model)).tobytes())
    for arr in (
        model.qpos0,
        model.body_pos,
        model.geom_pos,
        model.geom_size,
        model.tex_adr,
        model.mesh_vertadr,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _model() -> mj.MjModel:
    return mj.MjModel.from_xml_string(_XML)


class RenderModelSignatureTests(unittest.TestCase):
    # ---------------------------------------------------------------- stability

    def test_identical_models_share_a_key(self) -> None:
        """No false invalidation: the whole 87.5%-ceiling reuse scheme dies if
        two independently compiled copies of the same scene disagree."""
        self.assertEqual(render_model_signature(_model()), render_model_signature(_model()))

    def test_repeated_calls_on_one_model_are_stable(self) -> None:
        model = _model()
        self.assertEqual(render_model_signature(model), render_model_signature(model))

    def test_key_is_stable_across_processes(self) -> None:
        """The mjb dir is shared (often NFS) between engine workers, so a key
        that embeds anything process-local -- id(), PYTHONHASHSEED-sensitive
        hash(), dict order -- would drop reuse to zero and re-write hundreds of
        MB per env creation."""
        mine = render_model_signature(_model())
        script = textwrap.dedent(
            """
            import mujoco as mj
            from molmo_spaces.renderer.filament_rendering import (
                render_model_signature,
            )
            print(render_model_signature(mj.MjModel.from_xml_string(XML)))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", f"XML = {_XML!r}\n" + script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": "1234"},
        )
        self.assertEqual(out.stdout.strip(), mine)

    def test_hashes_live_contents_not_a_snapshot(self) -> None:
        """mjModel arrays are views into the model struct; an in-place write by
        a randomizer must be visible to the very next call."""
        model = _model()
        before = render_model_signature(model)
        model.geom_rgba[0][0] = 0.123
        self.assertNotEqual(render_model_signature(model), before)
        model.geom_rgba[0][0] = 1.0
        self.assertEqual(render_model_signature(model), before)

    # ------------------------------------------------------------- appearance

    def _assert_appearance_mutation_changes_key(self, mutate, label: str) -> None:
        base, mutated = _model(), _model()
        mutate(mutated)
        self.assertNotEqual(
            render_model_signature(base),
            render_model_signature(mutated),
            f"{label}: appearance change did not move the mjb content key",
        )
        # ...and confirm the old key really was blind to it, i.e. this test
        # would have failed before the fix.
        self.assertEqual(
            _legacy_signature(base),
            _legacy_signature(mutated),
            f"{label}: legacy key unexpectedly differs; test no longer pins B1",
        )

    def test_geom_rgba_changes_key(self) -> None:
        def mutate(m):
            m.geom_rgba[0] = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)

        self._assert_appearance_mutation_changes_key(mutate, "geom_rgba")

    def test_mat_rgba_changes_key(self) -> None:
        def mutate(m):
            m.mat_rgba[0] = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)

        self._assert_appearance_mutation_changes_key(mutate, "mat_rgba")

    def test_mat_scalar_properties_change_key(self) -> None:
        for field in (
            "mat_specular",
            "mat_shininess",
            "mat_emission",
            "mat_reflectance",
            "mat_metallic",
            "mat_roughness",
        ):
            with self.subTest(field=field):

                def mutate(m, field=field):
                    getattr(m, field)[0] = 0.77

                self._assert_appearance_mutation_changes_key(mutate, field)

    def test_mat_texid_change_key(self) -> None:
        def mutate(m):
            # Swap which texture material 0 points at -- same bytes in
            # tex_data, entirely different render.
            m.mat_texid[0] = m.mat_texid[1]

        self._assert_appearance_mutation_changes_key(mutate, "mat_texid")

    def test_tex_data_changes_key(self) -> None:
        def mutate(m):
            adr = int(m.tex_adr[0])
            m.tex_data[adr : adr + 32] = 7

        self._assert_appearance_mutation_changes_key(mutate, "tex_data")

    def test_tex_data_single_byte_change_key(self) -> None:
        """A one-byte perturbation deep inside a texture must move the key.

        This is the case any strided-sample or shape-only digest would miss,
        and it is why tex_data is hashed in full.
        """
        base, mutated = _model(), _model()
        adr = int(mutated.tex_adr[1])
        middle = adr + (int(mutated.tex_height[1]) * int(mutated.tex_width[1])) // 2 + 1
        mutated.tex_data[middle] = np.uint8(int(mutated.tex_data[middle]) ^ 0xFF)
        self.assertNotEqual(render_model_signature(base), render_model_signature(mutated))

    def test_light_fields_change_key(self) -> None:
        for field, value in (
            ("light_pos", 0.5),
            ("light_dir", 0.5),
            ("light_diffuse", 0.5),
            ("light_specular", 0.5),
            ("light_ambient", 0.5),
            ("light_attenuation", 0.5),
            ("light_cutoff", 0.5),
            ("light_exponent", 0.5),
            ("light_intensity", 42.0),
            ("light_range", 3.0),
            ("light_bulbradius", 0.3),
            ("light_active", 0),
            ("light_castshadow", 0),
            # 0 is this model's default light type, so perturb to a value that
            # is genuinely different.
            ("light_type", 1),
        ):
            with self.subTest(field=field):

                def mutate(m, field=field, value=value):
                    arr = getattr(m, field)
                    arr[0] = value

                self._assert_appearance_mutation_changes_key(mutate, field)

    def test_headlight_change_key(self) -> None:
        def mutate(m):
            m.vis.headlight.diffuse[0] = 0.9

        self._assert_appearance_mutation_changes_key(mutate, "vis.headlight.diffuse")

    def test_env_light_intensity_numeric_changes_key(self) -> None:
        """FILAMENT_ATTR_ENV_LIGHT_INTENSITY rides in numeric_data."""

        def mutate(m):
            m.numeric_data[0] = 12345.0

        self._assert_appearance_mutation_changes_key(mutate, "numeric_data")

    # -------------------------------------------------------------- geometry

    def test_geom_quat_changes_key(self) -> None:
        """Orientation was missing from the geometry half of the key too."""

        def mutate(m):
            m.geom_quat[0] = np.array([0.707, 0.0, 0.707, 0.0])

        self._assert_appearance_mutation_changes_key(mutate, "geom_quat")

    def test_geom_pos_still_changes_key(self) -> None:
        base, mutated = _model(), _model()
        mutated.geom_pos[0][0] += 0.25
        self.assertNotEqual(render_model_signature(base), render_model_signature(mutated))
        self.assertNotEqual(_legacy_signature(base), _legacy_signature(mutated))

    # ------------------------------------------------------------------ guard

    def test_no_appearance_field_is_left_out(self) -> None:
        """Fail when a mujoco upgrade adds an appearance field we do not hash.

        B1 was a silent omission; the only durable defence is to enumerate the
        installed mjModel rather than trust a hand-written list to stay
        complete.
        """
        covered = set(_SIG_ARRAY_FIELDS)
        appearance = {
            name
            for name in dir(mj.MjModel)
            if not name.startswith("_")
            and (
                name.startswith(("mat_", "tex_", "light_"))
                or name.endswith(("_rgba", "_matid", "_texid"))
            )
        }
        self.assertEqual(
            appearance - covered,
            set(),
            "mjModel appearance fields missing from the mjb content key",
        )
        # And the declared appearance list must not drift from what exists.
        self.assertEqual(set(_SIG_APPEARANCE_FIELDS) - set(dir(mj.MjModel)) - {
            "numeric_adr",
            "numeric_data",
            "numeric_size",
        }, set())


if __name__ == "__main__":
    unittest.main()
