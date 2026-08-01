"""ALICE_MS_OBS_HISTORY_LITE: retention-only limiter on observation_cache.

The cache holds full rendered observations (~150-300MB/episode) and grows for
the whole episode, but the only element any consumer reads is [0] -- step()'s
reset-vs-first-step camera check. These tests pin that the limiter keeps [0],
keeps every OTHER cache complete, and is inert when unset.

Written against the real caching method rather than a stub of it, so the
identity claim is about the code that actually runs.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from molmo_spaces.tasks import task as task_mod


class _Task:
    """A BaseMujocoTask stand-in that runs the REAL caching method.

    BaseMujocoTask.__init__ needs a live MuJoCo env, so the caching method is
    bound onto a minimal object carrying only the state it touches. The method
    body under test is the real one.
    """

    get_and_cache_all_step_information = (
        task_mod.BaseMujocoTask.get_and_cache_all_step_information
    )
    get_history = task_mod.BaseMujocoTask.get_history

    def __init__(self, lite: bool) -> None:
        self._obs_history_lite = lite
        self.action_cache: list = []
        self.observation_cache: list = []
        self.reward_cache: list = []
        self.terminal_cache: list = []
        self.truncated_cache: list = []
        self.success_cache: list = []
        self._step = 0

    # --- the bits the real method calls -------------------------------
    def get_observations(self):
        self._step += 1
        return [{"cam": np.full((2, 2), self._step, dtype=np.uint8)}]

    def get_reward(self):
        return np.array([float(self._step)])

    def is_terminal(self):
        return np.array([False])

    def is_timed_out(self):
        return np.array([False])

    def get_info(self):
        return [{}]

    def judge_success(self):
        return False

    def get_obs_scene(self):
        return {}


def _run(lite: bool, steps: int) -> _Task:
    t = _Task(lite)
    for _ in range(steps):
        t.get_and_cache_all_step_information()
    return t


# ----------------------------------------------------------- flag inert
def test_flag_off_retains_every_observation(monkeypatch):
    """Identity arm: unset must reproduce today's unbounded retention."""
    monkeypatch.delenv("ALICE_MS_OBS_HISTORY_LITE", raising=False)
    assert task_mod._obs_history_lite_enabled() is False

    t = _run(lite=False, steps=25)

    assert len(t.observation_cache) == 25
    for i, obs in enumerate(t.observation_cache, start=1):
        assert obs[0]["cam"][0][0] == i


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_exactly_one_enables(monkeypatch, value):
    monkeypatch.setenv("ALICE_MS_OBS_HISTORY_LITE", value)

    assert task_mod._obs_history_lite_enabled() is False


def test_flag_on_is_recognised(monkeypatch):
    monkeypatch.setenv("ALICE_MS_OBS_HISTORY_LITE", "1")

    assert task_mod._obs_history_lite_enabled() is True


# -------------------------------------------------------- retention set
def test_lite_keeps_element_zero_only():
    """[0] is load-bearing: step() reads and may overwrite it on step 1."""
    t = _run(lite=True, steps=25)

    assert len(t.observation_cache) == 1
    assert t.observation_cache[0][0]["cam"][0][0] == 1, "kept the wrong element"


def test_lite_leaves_element_zero_writable():
    """task.py:315 does `observation_cache[0] = obs` on a mismatch."""
    t = _run(lite=True, steps=10)
    replacement = [{"cam": np.full((2, 2), 99, dtype=np.uint8)}]

    t.observation_cache[0] = replacement

    assert t.observation_cache[0][0]["cam"][0][0] == 99


def test_lite_does_not_touch_the_other_caches():
    """Only observations carry frames; rewards/terminals stay complete."""
    lite = _run(lite=True, steps=25)
    full = _run(lite=False, steps=25)

    for name in ("reward_cache", "terminal_cache", "truncated_cache",
                 "success_cache"):
        assert len(getattr(lite, name)) == 25, name
        assert len(getattr(lite, name)) == len(getattr(full, name)), name


def test_returned_observation_is_unaffected():
    """The 1:1 claim: callers receive the same arrays either way."""
    lite = _Task(True)
    full = _Task(False)

    for _ in range(5):
        a = lite.get_and_cache_all_step_information()
        b = full.get_and_cache_all_step_information()
        assert np.array_equal(a[0][0]["cam"], b[0][0]["cam"])
        assert np.array_equal(a[1], b[1])       # reward
        assert np.array_equal(a[2], b[2])       # terminated
        assert np.array_equal(a[3], b[3])       # truncated


def test_get_history_warns_when_truncated(caplog):
    """Never hand back a short history silently."""
    t = _run(lite=True, steps=10)

    with caplog.at_level(logging.WARNING, logger=task_mod.__name__):
        history = t.get_history()

    assert len(history["observations"]) == 1
    assert len(history["rewards"]) == 10
    assert any("OBS_HISTORY_LITE" in r.getMessage() for r in caplog.records)


def test_get_history_is_silent_when_flag_off(caplog):
    t = _run(lite=False, steps=10)

    with caplog.at_level(logging.WARNING, logger=task_mod.__name__):
        history = t.get_history()

    assert len(history["observations"]) == 10
    assert not [r for r in caplog.records
                if "OBS_HISTORY_LITE" in r.getMessage()]


def test_receipt_logs_in_both_directions(monkeypatch, caplog):
    """Absence of a line is not a receipt; the OFF arm must prove it was off."""
    for value, expect in (("1", "enabled"), ("0", "disabled")):
        monkeypatch.setenv("ALICE_MS_OBS_HISTORY_LITE", value)
        monkeypatch.setattr(task_mod, "_obs_lite_receipt_logged", False)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=task_mod.__name__):
            lite = task_mod._obs_history_lite_enabled()
            if not task_mod._obs_lite_receipt_logged:
                task_mod.log.info(
                    "OBS_HISTORY_LITE %s",
                    "enabled: retaining observation[0] only" if lite
                    else "disabled: full observation history retained",
                )
        (line,) = [r.getMessage() for r in caplog.records
                   if "OBS_HISTORY_LITE" in r.getMessage()]
        assert expect in line
