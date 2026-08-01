"""P1: the NEW-key line that makes the cross-engine texture union computable.

The union is a set-union across engines of what each rebuild reports as new, so
the only properties that matter are (a) the same content produces the same id in
a different process, (b) different content does not, and (c) an over-cap rebuild
degrades predictably instead of emitting an arbitrary subset.
"""

import logging

import numpy as np
import pytest

from molmo_spaces.renderer import filament_rendering as fr


@pytest.fixture(autouse=True)
def fresh_process_cache():
    """Each test starts as a freshly spawned engine would."""
    saved = set(fr._PROCESS_TEXTURE_KEYS)
    fr._PROCESS_TEXTURE_KEYS.clear()
    yield
    fr._PROCESS_TEXTURE_KEYS.clear()
    fr._PROCESS_TEXTURE_KEYS.update(saved)


def _keys_line(caplog):
    lines = [
        r.getMessage()
        for r in caplog.records
        if "MS_FILAMENT_TEXTURE_CACHE_KEYS" in r.getMessage()
    ]
    assert len(lines) == 1, f"expected exactly one keys line, got {len(lines)}"
    return lines[0]


def _field(line, name):
    for part in line.split():
        if part.startswith(f"{name}="):
            return part[len(name) + 1:]
    raise AssertionError(f"{name}= missing from {line!r}")


# ------------------------------------------------------- fingerprint identity
def test_same_content_yields_the_same_id_in_another_process(caplog):
    """The whole union premise: engine A and engine B agree on an id."""
    keys = {"path:/assets/wood.png", "data:type=2:shape=8x8x3:" + "ab" * 16}

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(set(keys))
    first = _field(_keys_line(caplog), "keys")

    caplog.clear()
    fr._PROCESS_TEXTURE_KEYS.clear()  # a different engine, same textures
    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(set(keys))
    second = _field(_keys_line(caplog), "keys")

    assert first == second


def test_ids_are_eight_hex_chars():
    fp = fr._texture_fingerprint("path:/assets/wood.png")

    assert len(fp) == 8
    assert all(c in "0123456789abcdef" for c in fp)


def test_keys_sharing_a_long_prefix_do_not_collide():
    """Regression guard for the rejected design.

    A literal 8-character prefix of the key would map every path: texture onto
    "path:/da" and every data: texture onto "data:typ", shrinking the union and
    inflating the dedup ceiling that decides D1.
    """
    a = "path:/data/assets/textures/wood_floor_01.png"
    b = "path:/data/assets/textures/wood_floor_02.png"
    c = "data:type=2:colorspace=1:shape=1024x1024x3:" + "00" * 15 + "01"
    d = "data:type=2:colorspace=1:shape=1024x1024x3:" + "00" * 15 + "02"

    assert len({fr._texture_fingerprint(k) for k in (a, b, c, d)}) == 4


def test_nothing_new_emits_no_line(caplog):
    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(set())

    assert not [
        r for r in caplog.records if "MS_FILAMENT_TEXTURE_CACHE_KEYS" in r.getMessage()
    ]


# ------------------------------------------------------------------- line cap
def test_line_caps_and_declares_the_truncation(caplog):
    keys = {f"path:/t/{i}.png" for i in range(fr._TEXTURE_KEY_LOG_CAP + 150)}

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(keys)
    line = _keys_line(caplog)

    assert _field(line, "new_unique") == str(fr._TEXTURE_KEY_LOG_CAP + 150)
    assert _field(line, "emitted") == str(fr._TEXTURE_KEY_LOG_CAP)
    assert len(_field(line, "keys").split(",")) == fr._TEXTURE_KEY_LOG_CAP
    assert line.endswith(" truncated")


def test_under_the_cap_is_not_marked_truncated(caplog):
    keys = {f"path:/t/{i}.png" for i in range(70)}  # a normal rebuild

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(keys)
    line = _keys_line(caplog)

    assert _field(line, "new_unique") == "70"
    assert _field(line, "emitted") == "70"
    assert "truncated" not in line


def test_truncation_keeps_the_same_low_region_everywhere(caplog):
    """Over-cap lines must still be comparable across engines.

    Sorted ids mean every engine emits the lowest region of the SAME uniform
    hash space, so two engines holding a texture in common either both report
    it or both drop it -- an arbitrary subset would bias the union instead.
    """
    keys = {f"path:/t/{i}.png" for i in range(fr._TEXTURE_KEY_LOG_CAP + 150)}

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_new_texture_keys(set(keys))
    emitted = _field(_keys_line(caplog), "keys").split(",")

    expected = sorted(fr._texture_fingerprint(k) for k in keys)
    assert emitted == expected[: fr._TEXTURE_KEY_LOG_CAP]
    assert emitted == sorted(emitted)
    shared = {fr._texture_fingerprint("path:/t/7.png")}
    assert (shared <= set(emitted)) == (shared <= set(expected[: fr._TEXTURE_KEY_LOG_CAP]))


# ----------------------------------------------------- wired to the instrument
class _FakeModel:
    """Only the fields _texture_key reads, so no MjModel build is needed."""

    def __init__(self, paths, data_textures):
        blob = b""
        self.tex_pathadr = []
        for p in paths:
            self.tex_pathadr.append(len(blob))
            blob += p.encode() + b"\x00"
        self.paths = np.frombuffer(blob, dtype=np.uint8)
        self.tex_pathadr.extend([-1] * len(data_textures))

        self.tex_data = (
            np.concatenate([np.frombuffer(t, dtype=np.uint8) for t in data_textures])
            if data_textures
            else np.zeros(0, dtype=np.uint8)
        )
        self.tex_adr, adr = [], 0
        for _ in paths:
            self.tex_adr.append(0)
        for t in data_textures:
            self.tex_adr.append(adr)
            adr += len(t)

        n = len(paths) + len(data_textures)
        self.ntex = n
        sizes = [1] * len(paths) + [len(t) for t in data_textures]
        self.tex_height = [1] * n
        self.tex_width = sizes
        self.tex_nchannel = [1] * n
        self.tex_type = [2] * n
        self.tex_colorspace = [1] * n


def test_instrument_emits_both_lines_and_they_agree(caplog):
    model = _FakeModel(
        ["/assets/a.png", "/assets/b.png"],
        [b"\x01\x02\x03", b"\x04\x05\x06"],
    )

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_texture_cache_potential(model)

    parent = [
        r.getMessage()
        for r in caplog.records
        if "MS_FILAMENT_TEXTURE_CACHE_POTENTIAL" in r.getMessage()
    ]
    assert len(parent) == 1
    line = _keys_line(caplog)

    assert _field(parent[0], "process_new_unique") == _field(line, "new_unique") == "4"
    assert len(_field(line, "keys").split(",")) == 4


def test_second_upload_of_the_same_model_adds_nothing(caplog):
    """Within ONE process the keys are already known, so there is no new-key
    line -- the parent line still reports the hits."""
    model = _FakeModel(["/assets/a.png"], [b"\x01\x02\x03"])
    fr._log_texture_cache_potential(model)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_texture_cache_potential(model)

    assert not [
        r for r in caplog.records if "MS_FILAMENT_TEXTURE_CACHE_KEYS" in r.getMessage()
    ]
    parent = [
        r.getMessage()
        for r in caplog.records
        if "MS_FILAMENT_TEXTURE_CACHE_POTENTIAL" in r.getMessage()
    ][0]
    assert _field(parent, "process_new_unique") == "0"
    assert _field(parent, "process_seen_hits") == "2"


def test_disabled_instrument_emits_neither_line(caplog, monkeypatch):
    monkeypatch.setenv("ALICE_MS_FIL_TEXTURE_CACHE_LOG", "0")
    model = _FakeModel(["/assets/a.png"], [])

    with caplog.at_level(logging.INFO, logger=fr.log.name):
        fr._log_texture_cache_potential(model)

    assert not caplog.records
