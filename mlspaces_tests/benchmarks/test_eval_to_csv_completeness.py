"""A success rate must be a fraction of the benchmark, not of whatever files exist.

Filesystem-only: no MuJoCo, no GPU.
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_to_csv",
    Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "eval_to_csv.py",
)
eval_to_csv_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_to_csv_mod)

eval_to_csv = eval_to_csv_mod.eval_to_csv
CompletenessPolicy = eval_to_csv_mod.CompletenessPolicy
IncompleteEvalError = eval_to_csv_mod.IncompleteEvalError

T = 5


def _qpos_rows(n_steps: int) -> np.ndarray:
    """qpos is stored as the JSON-encoded byte rows _decode_json_sequence expects."""
    rows = []
    for step in range(n_steps):
        blob = json.dumps({"joints": [float(step)] * 7}).encode("utf-8")
        padded = np.zeros(2000, dtype=np.uint8)
        padded[: len(blob)] = list(blob)
        rows.append(padded)
    return np.stack(rows)


def _write_episode(run_dir: Path, index: int, *, success: bool = False) -> Path:
    ep_dir = run_dir / f"ep_{index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    path = ep_dir / f"trajectories_{index:06d}.h5"
    with h5py.File(path, "w") as fh:
        traj = fh.create_group("traj_0")
        arr = np.zeros(T, dtype=bool)
        arr[-1] = success
        traj.create_dataset("success", data=arr)
        traj.create_dataset("obs_scene", data=json.dumps({"object_name": "apple_01"}))
        traj.create_group("obs").create_group("agent").create_dataset("qpos", data=_qpos_rows(T))
    return path


def _write_manifest(run_dir: Path, num_episodes: int) -> None:
    (run_dir / "_MANIFEST.json").write_text(
        json.dumps({"num_episodes": num_episodes, "sources": []})
    )


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


def test_complete_set_scores(run_dir, tmp_path):
    _write_manifest(run_dir, 3)
    for i in range(3):
        _write_episode(run_dir, i, success=(i == 0))

    out = tmp_path / "results.csv"
    df = eval_to_csv(str(run_dir), "col9", output_csv=str(out))

    overall = df[df["category"] == "OVERALL"].iloc[0]
    assert overall["total"] == 3
    assert overall["successes"] == 1
    assert bool(overall["complete"]) is True


def test_incomplete_eval_raises(run_dir, tmp_path):
    """The headline case: today this returns a clean 7-episode rate and no signal."""
    _write_manifest(run_dir, 10)
    for i in range(7):
        _write_episode(run_dir, i)

    with pytest.raises(IncompleteEvalError) as excinfo:
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))

    message = str(excinfo.value)
    assert "expected 10 episodes, found 7" in message
    assert "[7, 8, 9]" in message
    assert not (tmp_path / "results.csv").exists()


def test_allow_incomplete_stamps_every_row(run_dir, tmp_path):
    _write_manifest(run_dir, 10)
    for i in range(7):
        _write_episode(run_dir, i)

    out = tmp_path / "results.csv"
    df = eval_to_csv(
        str(run_dir),
        "col9",
        output_csv=str(out),
        completeness=CompletenessPolicy(allow_incomplete=True),
    )

    assert not df["complete"].any()
    header = out.read_text().splitlines()
    assert any(line.startswith("# INCOMPLETE: 3 missing") for line in header)
    assert "# expected_episodes: 10" in header
    # The column must survive a comment-stripping concat into a board sheet.
    assert not pd.read_csv(out, comment="#")["complete"].any()


def test_partial_dir_is_not_scored(run_dir, tmp_path):
    """A crashed write leaves .partial; it is evidence, never an artifact."""
    _write_manifest(run_dir, 3)
    for i in range(3):
        _write_episode(run_dir, i)
    stale = run_dir / "ep_000003.partial"
    stale.mkdir()
    _write_episode(stale, 3)

    with pytest.raises(IncompleteEvalError, match=r"\.partial dirs:   1"):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))


def test_failed_marker_reason_reaches_the_operator(run_dir, tmp_path):
    _write_manifest(run_dir, 4)
    for i in range(3):
        _write_episode(run_dir, i)
    failed = run_dir / "_FAILED"
    failed.mkdir()
    (failed / "ep_000003.json").write_text(
        json.dumps({"episode_index": 3, "reason": "OSError(28, 'No space left on device')"})
    )

    with pytest.raises(IncompleteEvalError, match="No space left on device"):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))


def test_missing_success_array_is_not_dropped_silently(run_dir, tmp_path):
    _write_manifest(run_dir, 2)
    _write_episode(run_dir, 0)
    ep_dir = run_dir / "ep_000001"
    ep_dir.mkdir()
    with h5py.File(ep_dir / "trajectories_000001.h5", "w") as fh:
        fh.create_group("traj_0").create_dataset("obs_scene", data="{}")

    with pytest.raises(IncompleteEvalError, match="no `success` array"):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))


def test_unreadable_h5_is_not_skipped_silently(run_dir, tmp_path):
    _write_manifest(run_dir, 2)
    _write_episode(run_dir, 0)
    ep_dir = run_dir / "ep_000001"
    ep_dir.mkdir()
    (ep_dir / "trajectories_000001.h5").write_text("not an h5 file")

    with pytest.raises(IncompleteEvalError, match="unreadable trajectory h5"):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))


def test_expected_episodes_must_agree_with_manifest(run_dir, tmp_path):
    _write_manifest(run_dir, 10)
    _write_episode(run_dir, 0)

    with pytest.raises(IncompleteEvalError, match="disagrees with"):
        eval_to_csv(
            str(run_dir),
            "col9",
            output_csv=str(tmp_path / "results.csv"),
            completeness=CompletenessPolicy(expected_episodes=12),
        )


def test_no_manifest_warns_and_still_scores(run_dir, tmp_path, capsys):
    """Foreign run dirs (upstream's own docs invoke this) keep working, loudly."""
    for i in range(3):
        _write_episode(run_dir, i)

    out = tmp_path / "results.csv"
    df = eval_to_csv(str(run_dir), "col9", output_csv=str(out))

    assert df[df["category"] == "OVERALL"].iloc[0]["total"] == 3
    assert df["complete"].isna().all()
    assert "cannot be trusted as a board number" in capsys.readouterr().err


def test_scoring_zero_episodes_refuses(run_dir, tmp_path):
    _write_manifest(run_dir, 1)
    ep_dir = run_dir / "ep_000000"
    ep_dir.mkdir()
    with h5py.File(ep_dir / "trajectories_000000.h5", "w") as fh:
        fh.create_group("traj_0").create_dataset("obs_scene", data="{}")

    with pytest.raises(IncompleteEvalError):
        eval_to_csv(
            str(run_dir),
            "col9",
            output_csv=str(tmp_path / "results.csv"),
            completeness=CompletenessPolicy(allow_incomplete=False),
        )


def test_no_temp_files_leak_on_refusal(run_dir, tmp_path):
    _write_manifest(run_dir, 10)
    for i in range(7):
        _write_episode(run_dir, i)
    # tempfile honours $TMPDIR; hardcoding /tmp silently disarms this guard.
    tmp_root = tempfile.gettempdir()
    before = set(os.listdir(tmp_root))

    with pytest.raises(IncompleteEvalError):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))

    leaked = [n for n in set(os.listdir(tmp_root)) - before if n.endswith(".h5")]
    assert leaked == []


def test_scored_count_must_reach_expected(run_dir, tmp_path):
    """complete=True must not sit on a short denominator.

    Every expected ep_NNNNNN/ exists, but one publishes a traj group the scorer cannot
    count. Before the fix the filesystem survey saw 3/3 and reported complete=True.
    """
    _write_manifest(run_dir, 3)
    for i in range(2):
        _write_episode(run_dir, i)
    ep_dir = run_dir / "ep_000002"
    ep_dir.mkdir()
    with h5py.File(ep_dir / "trajectories_000002.h5", "w") as fh:
        fh.create_group("not_a_traj_group")

    with pytest.raises(IncompleteEvalError):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))


def test_failure_evidence_without_manifest_still_refuses(run_dir, tmp_path):
    """A _FAILED marker means an episode is KNOWN lost, manifest or not.

    Before the fix the expected-is-None branch returned above this check, so a run with
    durable failure evidence and no manifest exited 0 with a clean-looking rate.
    """
    for i in range(3):
        _write_episode(run_dir, i)
    failed = run_dir / "_FAILED"
    failed.mkdir()
    (failed / "ep_000003.json").write_text(
        json.dumps({"episode_index": 3, "reason": "OSError(28, 'No space left')"})
    )

    with pytest.raises(IncompleteEvalError, match="known lost"):
        eval_to_csv(str(run_dir), "col9", output_csv=str(tmp_path / "results.csv"))

    # --allow-incomplete still downgrades it to a loud warning.
    df = eval_to_csv(
        str(run_dir),
        "col9",
        output_csv=str(tmp_path / "r2.csv"),
        completeness=CompletenessPolicy(allow_incomplete=True),
    )
    assert df[df["category"] == "OVERALL"].iloc[0]["total"] == 3
