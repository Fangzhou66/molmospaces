from pathlib import Path

from molmo_spaces.tasks import task_sampler


def _patch_cache_roots(monkeypatch, tmp_path):
    versions = {
        "objects": {
            "objaverse": "20260131",
            "thor": "20251117",
        },
        "robots": {},
        "scenes": {
            "procthor-objaverse-val": "20251205_with_occupancy",
        },
    }
    monkeypatch.setattr(task_sampler, "DATA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(task_sampler, "ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr(task_sampler, "DATA_TYPE_TO_SOURCE_TO_VERSION", versions)
    task_sampler._thor_cache_suffix_candidates.cache_clear()


def test_render_service_xml_resolves_objaverse_assets_from_bad_scene_path(
    monkeypatch, tmp_path
):
    _patch_cache_roots(monkeypatch, tmp_path)
    uid = "660bfdd0e9144271acdc116632297395"
    scene_dir = (
        task_sampler.DATA_CACHE_DIR
        / "scenes"
        / "procthor-objaverse-val"
        / "20251205_with_occupancy"
    )
    expected = (
        task_sampler.DATA_CACHE_DIR
        / "objects"
        / "objaverse"
        / "20260131"
        / uid
        / f"{uid}_visual.obj"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text("# obj\n")

    resolved = task_sampler._resolve_render_service_xml_path(
        scene_dir / f"{uid}_visual.obj",
        scene_dir,
    )

    assert resolved == str(expected.resolve())


def test_render_service_xml_resolves_thor_assets_from_suffix(monkeypatch, tmp_path):
    _patch_cache_roots(monkeypatch, tmp_path)
    scene_dir = (
        task_sampler.DATA_CACHE_DIR
        / "scenes"
        / "procthor-objaverse-val"
        / "20251205_with_occupancy"
    )
    expected = (
        task_sampler.DATA_CACHE_DIR
        / "objects"
        / "thor"
        / "20251117"
        / "Kitchen Objects"
        / "Bowl"
        / "Prefabs"
        / "Bowl_29"
        / "Bowl_29_bowl_29"
        / "Bowl_29_bowl_29_0.obj"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text("# obj\n")

    resolved = task_sampler._resolve_render_service_xml_path(
        scene_dir / "Bowl_29_bowl_29" / "Bowl_29_bowl_29_0.obj",
        scene_dir,
    )

    assert resolved == str(expected.resolve())
