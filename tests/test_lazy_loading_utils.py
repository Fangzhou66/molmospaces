from pathlib import Path

from molmo_spaces.utils import lazy_loading_utils


class _ResourceManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def source_dir(self, data_type: str, source: str) -> Path:
        assert data_type == "objects"
        assert source == "objaverse"
        return self.cache_dir / "objects" / "objaverse" / "20260131"


def test_find_object_paths_accepts_symlinked_cache_root(monkeypatch, tmp_path):
    real_cache = tmp_path / "data_users_cache" / "molmo-spaces-resources"
    link_cache = tmp_path / "home_cache" / "molmo-spaces-resources"
    link_cache.parent.mkdir(parents=True)
    link_cache.symlink_to(real_cache, target_is_directory=True)

    monkeypatch.setattr(
        lazy_loading_utils,
        "get_resource_manager",
        lambda: _ResourceManager(link_cache),
    )

    uid = "110a5fc31b1f459cb4e05b50b092af1b"
    object_path = (
        real_cache
        / "objects"
        / "objaverse"
        / "20260131"
        / uid
        / f"{uid}_visual.obj"
    )
    object_path.parent.mkdir(parents=True)
    object_path.write_text("# obj\n")

    scene_dir = real_cache / "scenes" / "procthor-objaverse-val" / "20251205_with_occupancy"
    scene_dir.mkdir(parents=True)
    scene_path = scene_dir / "scene.xml"
    scene_path.write_text(
        f"""
<mujoco>
  <asset>
    <mesh file="../../../objects/objaverse/20260131/{uid}/{uid}_visual.obj"/>
  </asset>
</mujoco>
""".strip()
    )

    assert list(lazy_loading_utils.find_object_paths(scene_path)) == [
        ("objaverse", Path(uid) / f"{uid}_visual.obj")
    ]
