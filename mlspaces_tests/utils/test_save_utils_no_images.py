"""The scored h5 must never carry image tensors, for any caller or camera count."""

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from molmo_spaces.env.abstract_sensors import Sensor, SensorSuite
from molmo_spaces.env.sensors_cameras import CameraSensor
from molmo_spaces.utils.save_utils import (
    is_camera_sensor,
    prepare_episode_for_saving,
    save_trajectories,
)

# The five cameras the scored FrankaPickandPlaceHardBench columns actually record.
# Only one carries a "camera" token, which is why detection here is shape- or
# suite-based and never name-based.
SCORED_CAMERAS = [
    "wrist_camera_zed_mini",
    "droid_shoulder_light_randomization",
    "randomized_zed2_analogue_1",
    "randomized_zed2_analogue_2",
    "randomized_gopro_analogue_1",
]
RES = (624, 352)  # (width, height), the production benchmark resolution
T = 8


class _ArraySensor(Sensor):
    """Minimal non-camera sensor holding a fixed-shape float array."""

    def __init__(self, uuid: str, shape: tuple[int, ...]) -> None:
        import gymnasium.spaces as gyms

        super().__init__(uuid=uuid, observation_space=gyms.Box(-1.0, 1.0, shape=shape))

    def get_observation(self, env, task, *args, **kwargs):
        raise NotImplementedError


def _suite(n_cams: int) -> SensorSuite:
    sensors: list[Sensor] = [
        CameraSensor(camera_name=name, img_resolution=RES, uuid=name)
        for name in SCORED_CAMERAS[:n_cams]
    ]
    sensors.append(_ArraySensor("qpos", (9,)))
    sensors.append(_ArraySensor("tcp_pose", (7,)))
    return SensorSuite(sensors)


def _history(n_cams: int) -> dict:
    width, height = RES
    observations = []
    for t in range(T):
        obs = {
            name: np.full((height, width, 3), t, dtype=np.uint8)
            for name in SCORED_CAMERAS[:n_cams]
        }
        obs["qpos"] = np.arange(9, dtype=np.float32) + t
        obs["tcp_pose"] = np.arange(7, dtype=np.float32) + t
        observations.append([obs])
    return {
        "observations": observations,
        "rewards": [[float(t)] for t in range(T)],
        "terminals": [[t == T - 1] for t in range(T)],
        "truncateds": [[False] for _ in range(T)],
        "successes": [[t == T - 1] for t in range(T)],
        "obs_scene": {"scene": "fixture"},
    }


def _image_like_datasets(path) -> dict[str, tuple]:
    """Every dataset in the h5 whose shape looks like a stack of images."""
    found: dict[str, tuple] = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            if obj.shape[1] >= 32 and obj.shape[2] >= 32:
                found[name] = obj.shape

    with h5py.File(path, "r") as fh:
        fh.visititems(visit)
    return found


@pytest.mark.parametrize("n_cams", [1, 2, 5])
def test_scored_h5_never_contains_images(tmp_path, n_cams):
    """The eval path (save_dir=None, save_mp4s=False) must drop every frame.

    Parametrised over camera count on purpose: the invariant belongs to the write
    path, not to how many cameras a column happens to prune down to.
    """
    suite = _suite(n_cams)
    prepared = prepare_episode_for_saving(_history(n_cams), suite, fps=10.0, save_dir=None)

    assert prepared is not None
    for name in SCORED_CAMERAS[:n_cams]:
        assert name not in prepared, f"{name} survived into the batched episode dict"

    out = tmp_path / "ep"
    save_trajectories([prepared], save_dir=str(out), fps=10.0, save_mp4s=False)
    h5_path = out / "trajectories.h5"

    assert _image_like_datasets(h5_path) == {}
    assert h5_path.stat().st_size < 1_000_000

    # The three fields eval_to_csv actually reads must survive untouched.
    with h5py.File(h5_path, "r") as fh:
        assert len(fh["traj_0/success"]) == T
        assert fh["traj_0/obs/agent/qpos"].shape == (T, 9)
        assert json.loads(fh["traj_0/obs_scene"][()]) == {"scene": "fixture"}


def test_save_trajectories_refuses_image_tensors(tmp_path):
    """A batched dict still holding frames must fail before the file is created."""
    import torch

    width, height = RES
    prepared = {
        "qpos": torch.zeros(T, 9),
        "wrist_camera_zed_mini": torch.zeros(T, height, width, 3, dtype=torch.uint8),
    }
    out = tmp_path / "ep"
    out.mkdir()

    with pytest.raises(RuntimeError, match="Refusing to write image tensors"):
        save_trajectories([prepared], save_dir=str(out), fps=10.0, save_mp4s=False)

    assert list(out.iterdir()) == []


def test_is_camera_sensor_requires_suite():
    """No suite means no answer -- a name heuristic would misclassify 4 of the 5."""
    with pytest.raises(ValueError, match="without a SensorSuite"):
        is_camera_sensor("droid_shoulder_light_randomization")

    suite = _suite(2)
    assert is_camera_sensor("qpos", suite) is False
    assert is_camera_sensor("wrist_camera_zed_mini", suite) is True
    assert is_camera_sensor("not_a_member", suite) is False


def test_datagen_path_writes_videos_and_matching_h5(tmp_path):
    """save_dir set (the datagen posture) is structurally unchanged by the de-indent."""
    suite = _suite(2)
    out = tmp_path / "ep"
    out.mkdir()
    prepared = prepare_episode_for_saving(_history(2), suite, fps=10.0, save_dir=str(out))

    assert prepared is not None
    for name in SCORED_CAMERAS[:2]:
        assert name not in prepared

    mp4s = sorted(p.name for p in out.glob("*.mp4"))
    assert len(mp4s) == 2, mp4s

    save_trajectories([prepared], save_dir=str(out), fps=10.0, save_mp4s=True)
    assert _image_like_datasets(out / "trajectories.h5") == {}


def test_strip_is_independent_of_save_dir():
    """The frames leave the observation dicts whether or not videos were written."""
    without = prepare_episode_for_saving(_history(5), _suite(5), fps=10.0, save_dir=None)
    assert without is not None
    assert set(without) == {"qpos", "tcp_pose", "rewards", "terminals", "truncateds",
                            "successes", "obs_scene"}


def test_prepare_episode_for_saving_keeps_frames_when_opted_out():
    """remove_camera_sensors=False is the only way to retain frames, and no caller uses it."""
    prepared = prepare_episode_for_saving(
        _history(1), _suite(1), fps=10.0, save_dir=None, remove_camera_sensors=False
    )
    assert prepared is not None
    assert prepared["wrist_camera_zed_mini"].shape == (T, RES[1], RES[0], 3)


def test_save_utils_imports_without_engine():
    """save_utils must stay importable without a GPU, a display, or MuJoCo state."""
    proc = subprocess.run(
        [sys.executable, "-c", "import molmo_spaces.utils.save_utils"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
