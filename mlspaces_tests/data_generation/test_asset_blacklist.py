"""The blacklist a scored run reads is a pin, not a scratchpad.

It decides which bodies _delete_blacklisted_bodies removes before scene compile, so
a run that appends to it mid-run scores later episodes in a different environment
than earlier ones. Filesystem-only: no MuJoCo, no GPU.
"""

import importlib

import pytest

from molmo_spaces.tasks import task_sampler as ts

UPSTREAM_UIDS = [
    "a3d6f7df9ff94ed59f95d5086d5f3fdd",
    "87bb73dea4a341e5b5ead7543881f47d",
    "8bde878be8dd4fb7b6914db9c4dd31bb",
    "bf8ace08150545fb92f037f887658c37",
]


@pytest.fixture
def fresh(monkeypatch):
    """A module whose blacklist state is not shared with the rest of the suite."""
    mod = importlib.reload(ts)
    monkeypatch.setattr(mod, "_STATIC_ASSET_BLACKLIST", None, raising=False)
    monkeypatch.setattr(mod, "_BLACKLIST_SEALED", False, raising=False)
    return mod


def _write_blacklist(path, uids):
    path.write_text("# header comment\n\n" + "".join(f"{u}  # why\n" for u in uids))


def test_repo_blacklist_is_pinned_to_upstream(fresh):
    """Red the moment a runtime discovery is committed into source."""
    blacklist = fresh.load_asset_blacklist()
    assert blacklist == set(UPSTREAM_UIDS)
    assert fresh.asset_blacklist_digest(blacklist) == fresh.UPSTREAM_ASSET_BLACKLIST_SHA256


def test_digest_ignores_comments_and_order(fresh, tmp_path, monkeypatch):
    path = tmp_path / "asset_blacklist.txt"
    _write_blacklist(path, list(reversed(UPSTREAM_UIDS)))
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(path))

    assert fresh.asset_blacklist_digest(fresh.load_asset_blacklist()) == (
        fresh.UPSTREAM_ASSET_BLACKLIST_SHA256
    )


def test_pin_fails_loud_on_drift(fresh, tmp_path, monkeypatch):
    path = tmp_path / "asset_blacklist.txt"
    _write_blacklist(path, [*UPSTREAM_UIDS, "deadbeefdeadbeefdeadbeefdeadbeef"])
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(path))

    with pytest.raises(RuntimeError) as excinfo:
        fresh.assert_asset_blacklist_pinned()
    message = str(excinfo.value)
    assert str(path) in message
    assert fresh.UPSTREAM_ASSET_BLACKLIST_SHA256 in message

    # An explicit override is the documented escape, and it must work.
    fresh.assert_asset_blacklist_pinned(
        fresh.asset_blacklist_digest(fresh.get_static_asset_blacklist())
    )


def test_seal_blocks_persistence_and_in_process_mutation(fresh, tmp_path, monkeypatch):
    path = tmp_path / "asset_blacklist.txt"
    _write_blacklist(path, UPSTREAM_UIDS)
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(path))
    before = path.read_bytes()

    sealed = fresh.seal_asset_blacklist()
    assert sealed == set(UPSTREAM_UIDS)

    added = fresh.add_to_static_blacklist("cafebabecafebabecafebabecafebabe", "mass/inertia")

    assert added is False
    assert path.read_bytes() == before
    assert fresh.get_static_asset_blacklist() == set(UPSTREAM_UIDS)


def test_unsealed_datagen_does_not_write_into_the_repo(fresh, tmp_path, monkeypatch):
    """No write path configured -> self-heal in memory, never touch the source file."""
    path = tmp_path / "asset_blacklist.txt"
    _write_blacklist(path, UPSTREAM_UIDS)
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(path))
    monkeypatch.delenv("MLSPACES_ASSET_BLACKLIST_WRITE_PATH", raising=False)
    fresh.get_static_asset_blacklist()
    before = path.read_bytes()

    # MLSPACES_ASSET_BLACKLIST_PATH doubles as a write target by design, so the
    # repo-only case is the one where NEITHER var points outside the repo.
    monkeypatch.delenv("MLSPACES_ASSET_BLACKLIST_PATH", raising=False)
    added = fresh.add_to_static_blacklist("cafebabecafebabecafebabecafebabe", "mass/inertia")

    assert added is False
    assert path.read_bytes() == before
    assert "cafebabecafebabecafebabecafebabe" in fresh.get_static_asset_blacklist()


def test_unsealed_datagen_persists_to_the_write_path(fresh, tmp_path, monkeypatch):
    source = tmp_path / "asset_blacklist.txt"
    _write_blacklist(source, UPSTREAM_UIDS)
    runtime = tmp_path / "state" / "runtime_blacklist.txt"
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(source))
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_WRITE_PATH", str(runtime))

    assert fresh.add_to_static_blacklist("cafebabecafebabecafebabecafebabe", "mass/inertia")

    def runtime_uids():
        return [
            line.split("#")[0].strip()
            for line in runtime.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    assert runtime_uids() == ["cafebabecafebabecafebabecafebabe"]
    assert "cafebabe" not in source.read_text()
    # Second call must dedupe against the WRITE file, not the read file.
    assert fresh.add_to_static_blacklist("cafebabecafebabecafebabecafebabe", "again") is False
    assert runtime_uids() == ["cafebabecafebabecafebabecafebabe"]


def test_seal_is_idempotent(fresh, tmp_path, monkeypatch):
    path = tmp_path / "asset_blacklist.txt"
    _write_blacklist(path, UPSTREAM_UIDS)
    monkeypatch.setenv("MLSPACES_ASSET_BLACKLIST_PATH", str(path))

    assert fresh.seal_asset_blacklist() == fresh.seal_asset_blacklist()


def test_blacklisted_body_is_NOT_skipped():
    """A missing body means the scene does not match the benchmark: do not score it.

    Inverted deliberately. The caller re-raises when this returns None, which is what
    upstream does -- it has neither this function nor the blacklist branch.
    """
    # json_eval_task_sampler is only importable once the evaluation package is
    # initialised; importing it first hits a pre-existing circular import.
    importlib.import_module("molmo_spaces.evaluation.benchmark_schema")
    from molmo_spaces.tasks.json_eval_task_sampler import missing_object_pose_body_skip_reason

    reason = missing_object_pose_body_skip_reason(
        body_name=f"objaceilingpanel_{UPSTREAM_UIDS[0]}_1_0_2",
        selected_place_receptacle_name=None,
        static_asset_blacklist={UPSTREAM_UIDS[0]},
    )
    assert reason is None


def test_unselected_place_receptacle_is_still_skipped():
    """The other branch of the same function is live and must not regress."""
    # json_eval_task_sampler is only importable once the evaluation package is
    # initialised; importing it first hits a pre-existing circular import.
    importlib.import_module("molmo_spaces.evaluation.benchmark_schema")
    from molmo_spaces.tasks.json_eval_task_sampler import missing_object_pose_body_skip_reason

    reason = missing_object_pose_body_skip_reason(
        body_name="place_receptacle/candidate_b",
        selected_place_receptacle_name="place_receptacle/candidate_a",
    )
    assert reason == "unselected place_receptacle candidate"
