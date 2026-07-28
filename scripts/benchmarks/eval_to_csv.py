import os, json, tempfile
import argparse
import glob
import re
import sys
import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from scipy.stats import beta as beta_dist
import logging
log = logging.getLogger(__name__)

MANIFEST_NAME = "_MANIFEST.json"
FAILED_DIR_NAME = "_FAILED"
PARTIAL_SUFFIX = ".partial"


class IncompleteEvalError(RuntimeError):
    """The set of h5s on disk is not the set of episodes in the benchmark."""


@dataclass(frozen=True)
class CompletenessPolicy:
    """How to treat a scored set that does not match the benchmark.

    `expected_episodes` overrides the manifest; `allow_incomplete` downgrades the
    refusal to a stderr warning plus `complete=False` in every CSV row.
    """

    expected_episodes: int | None = None
    allow_incomplete: bool = False


@dataclass
class EpisodeSetReport:
    """What is on disk versus what the benchmark says should be."""

    expected: int | None
    found: set = field(default_factory=set)
    failed: list = field(default_factory=list)
    partial: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)
    # Episodes the scorer actually counted. Distinct from len(found): a published
    # ep_NNNNNN/ can still contribute no scored episode (empty traj group, a copy
    # that aborted mid-merge). Without this, `complete` could be True over a short
    # denominator -- the exact defect this class exists to prevent.
    scored: int | None = None

    @property
    def missing(self) -> list:
        if self.expected is None:
            return []
        return sorted(set(range(self.expected)) - self.found)

    @property
    def complete(self):
        """True/False, or None when the expected set is unknown."""
        if self.expected is None:
            return None
        if self.missing or self.failed or self.partial or self.unreadable:
            return False
        # The scored count must reach the benchmark's count, not merely the count of
        # directories on disk.
        return self.scored is None or self.scored == self.expected

    def describe(self, run_path) -> str:
        reasons = []
        for marker in self.failed[:20]:
            with open(marker) as fh:
                reasons.append(f"  {os.path.basename(marker)}: {json.load(fh)['reason']}")
        missing = self.missing
        return (
            f"INCOMPLETE eval under {run_path}: expected {self.expected} episodes, "
            f"found {len(self.found)}.\n"
            f"  missing indices ({len(missing)}): {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}\n"
            f"  {FAILED_DIR_NAME} markers: {len(self.failed)}\n"
            f"  {PARTIAL_SUFFIX} dirs:   {len(self.partial)}\n"
            f"  unreadable h5s:  {len(self.unreadable)}\n"
            + ("\n".join(reasons) if reasons else "")
        )


def _survey_episode_set(run_path, expected_episodes=None) -> EpisodeSetReport:
    """Reconcile the published episode dirs against the benchmark's own count.

    The denominator must come from the benchmark, not from whichever files happen
    to exist. `_MANIFEST.json` is written by the MolmoSpaces adapter, which is the
    only component that knows how many episodes the catalog holds.
    """
    manifest_path = os.path.join(run_path, MANIFEST_NAME)
    manifest_n = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest_n = int(json.load(fh)["num_episodes"])
    if expected_episodes is not None and manifest_n is not None and expected_episodes != manifest_n:
        raise IncompleteEvalError(
            f"--expected-episodes {expected_episodes} disagrees with {manifest_path} "
            f"({manifest_n})"
        )

    found = set()
    for entry in Path(run_path).iterdir():
        match = re.fullmatch(r"ep_(\d{6})", entry.name) if entry.is_dir() else None
        if match:
            found.add(int(match.group(1)))

    return EpisodeSetReport(
        expected=expected_episodes if expected_episodes is not None else manifest_n,
        found=found,
        failed=sorted(glob.glob(os.path.join(run_path, FAILED_DIR_NAME, "*.json"))),
        partial=sorted(glob.glob(os.path.join(run_path, f"ep_*{PARTIAL_SUFFIX}"))),
    )

THOR_CAT_SIMPLIFY = {
    "saltshaker": "S/P Shaker", "peppershaker": "S/P Shaker",
    "tomato": "Fruit", "apple": "Fruit",
    "butterknife": "Knife", "boiler": "Kettle",
    "winebottle": "Bottle", "atomizer": "Spray Bottle",
    "remotecontrol": "Remote Control", "soapdispenser": "Soap Dispenser",
    "tissuepaper": "Tissue Paper",
}

def get_success_any(success_array: np.ndarray) -> bool:
    """True if any of the elements of success_array are True."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(np.any(success_array))


def get_success_last_frame(success_array: np.ndarray) -> bool:
    """True iff the last element of success_array is True (current metric)."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(success_array[-1])


def _extract_object_name(obs_scene_bytes):
    try:
        obs = json.loads(obs_scene_bytes.decode("utf-8"))
        raw = obs.get("object_name", "unknown")
        cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
        return cleaned.split()[0] if cleaned else "unknown"
    except Exception:
        return "unknown"


def _simplify(name: str) -> str:
    simp = THOR_CAT_SIMPLIFY.get(name.lower(), name)
    return " ".join(w.capitalize() for w in simp.split())


def _bayesian_ci(successes, total, alpha=0.05):
    if total == 0:
        return 0.0, 0.0
    a, b = 1 + successes, 1 + (total - successes)
    return beta_dist.ppf(alpha / 2, a, b) * 100, beta_dist.ppf(1 - alpha / 2, a, b) * 100


def _copy_group(src, dst):
    for k, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(k, data=item[()])
        elif isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(k))


def _decode_json_sequence(raw_uint8):
    rows = []
    for row in raw_uint8:
        d = json.loads(bytes(row).rstrip(b"\x00").decode("utf-8"))
        flat = []
        for v in d.values():
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        rows.append(flat)
    return np.array(rows, dtype=np.float64)


def _episode_joint_jerk(ep, dt, max_steps=None):
    raw_q = None
    try:
        raw_q = ep["obs"]["agent"]["qpos"][:]
    except KeyError:
        try:
            raw_q = ep["actions"]["joint_pos"][:]
        except KeyError:
            pass
    if raw_q is None:
        return np.nan
    if max_steps is not None:
        raw_q = raw_q[:max_steps]
    q = _decode_json_sequence(raw_q)
    if q.shape[0] < 4:
        return np.nan
    d3 = q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]
    d3 /= dt ** 3
    return float(np.mean(np.linalg.norm(d3, axis=1)))


def _combine_trajectories(folder_path, *, allow_incomplete=False):
    """Merge the published per-episode h5s into one temp file.

    Returns (combined_path, unreadable_paths).
    """
    folder = Path(folder_path)
    # A crashed run leaves ep_NNNNNN.partial/ holding a half-written h5. Only the
    # renamed directory is a verified artifact, so .partial is never scored.
    h5_files = sorted(
        p for p in folder.rglob("*.h5")
        if not any(part.endswith(PARTIAL_SUFFIX) for part in p.parts)
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 found under {folder_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    out = h5py.File(tmp.name, "w")
    ep = 0
    unreadable = []
    for src_path in h5_files:
        try:
            src = h5py.File(src_path, "r")
            for tk in [k for k in src.keys() if k.startswith("traj_")]:
                name = f"episode_{ep:04d}_{tk}"
                dst = out.create_group(name)
                try:
                    _copy_group(src[tk], dst)
                except Exception:
                    # A copy that dies partway leaves a half-populated group AND an
                    # unincremented counter, so the next file collides on the same name
                    # and every later file is skipped -- silently truncating the
                    # denominator forward-only. Drop the stub so the name is free.
                    del out[name]
                    raise
                ep += 1
            src.close()
        except Exception as e:
            if not allow_incomplete:
                out.close()
                os.unlink(tmp.name)
                raise IncompleteEvalError(
                    f"unreadable trajectory h5 {src_path}: {e!r}; refusing to score a "
                    f"partial set (pass --allow-incomplete to override)"
                ) from e
            unreadable.append(str(src_path))
            print(f"Warning: skipping {src_path}: {e}", file=sys.stderr)
    out.close()
    print(f"Combined {ep} episodes from {len(h5_files)} files → {tmp.name}")
    return tmp.name, unreadable


def _build_row(policy_name, category, s, t, jerk_list, report_both, oracle_s=0):
    rate = 100.0 * s / t if t else 0.0
    ci_lo, ci_hi = _bayesian_ci(s, t)
    mean_jj = float(np.mean(jerk_list)) if jerk_list else np.nan
    std_jj = float(np.std(jerk_list)) if jerk_list else np.nan
    row = dict(
        policy=policy_name, category=category, successes=s, total=t,
        success_rate_pct=round(rate, 2),
        ci_95_low_pct=round(ci_lo, 2), ci_95_high_pct=round(ci_hi, 2),
    )
    if report_both:
        o_rate = 100.0 * oracle_s / t if t else 0.0
        o_ci_lo, o_ci_hi = _bayesian_ci(oracle_s, t)
        row.update(
            oracle_successes=oracle_s,
            oracle_rate_pct=round(o_rate, 2),
            oracle_ci_95_low_pct=round(o_ci_lo, 2),
            oracle_ci_95_high_pct=round(o_ci_hi, 2),
        )
    row.update(
        jerk_joint_mean=round(mean_jj, 6) if not np.isnan(mean_jj) else np.nan,
        jerk_joint_std=round(std_jj, 6) if not np.isnan(std_jj) else np.nan,
    )
    return row, rate


@dataclass(frozen=True)
class ScoringOptions:
    """How each episode's success array is reduced to a single verdict."""

    success_condition: str = "at-end"
    dt: float = 0.1
    max_steps: int | None = None
    allow_incomplete: bool = False

    @property
    def report_both(self) -> bool:
        return self.success_condition == "both"


@dataclass
class Tally:
    per_obj: dict = field(default_factory=dict)
    total_s: int = 0
    total_os: int = 0
    total_n: int = 0
    all_jerk_joint: list = field(default_factory=list)


def _episode_verdict(s_arr, opts):
    """(success, oracle_success) for one episode under the configured condition."""
    if opts.report_both:
        return get_success_last_frame(s_arr), get_success_any(s_arr)
    if opts.success_condition == "at-end":
        return get_success_last_frame(s_arr), None
    if opts.success_condition == "oracle":
        return get_success_any(s_arr), None
    raise ValueError(f"Unknown success condition: {opts.success_condition}")


def _score_combined(combined_h5, opts) -> Tally:
    """Aggregate per-object and overall counts from the merged h5."""
    per_obj = defaultdict(lambda: {"success": 0, "oracle_success": 0, "total": 0, "jerk_joint": []})
    tally = Tally(per_obj=per_obj)

    with h5py.File(combined_h5, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("episode_"):
                continue
            ep = f[key]

            if "success" not in ep:
                # This module configures no logging handler, so the log.info that
                # used to sit here produced no output at all -- the episode left the
                # denominator with no trace anywhere.
                if not opts.allow_incomplete:
                    raise IncompleteEvalError(
                        f"episode {key} has no `success` array; refusing to drop it "
                        f"silently from the denominator (pass --allow-incomplete)"
                    )
                print(f"Warning: no success array for {key}, skipping", file=sys.stderr)
                continue

            s_arr = ep["success"][:opts.max_steps] if opts.max_steps is not None else ep["success"][:]
            success, oracle_success = _episode_verdict(s_arr, opts)

            jj = _episode_joint_jerk(ep, opts.dt, max_steps=opts.max_steps)
            obj = _simplify(_extract_object_name(ep["obs_scene"][()])) if "obs_scene" in ep else "Unknown"

            per_obj[obj]["total"] += 1
            per_obj[obj]["success"] += int(success)
            if oracle_success is not None:
                per_obj[obj]["oracle_success"] += int(oracle_success)
            if not np.isnan(jj):
                per_obj[obj]["jerk_joint"].append(jj)
                tally.all_jerk_joint.append(jj)

            tally.total_n += 1
            tally.total_s += int(success)
            if oracle_success is not None:
                tally.total_os += int(oracle_success)

    return tally


def _enforce_completeness(report, run_path, *, allow_incomplete):
    """Refuse to emit a rate over a biased subset, or say loudly that it is one."""
    # Durable failure evidence is decisive even without a manifest: a _FAILED marker or
    # a .partial dir means an episode is KNOWN lost. Checking this before the
    # expected-is-None early return, which otherwise let such a run exit 0.
    if not report.expected and (report.failed or report.partial or report.unreadable):
        message = (
            f"INCOMPLETE eval under {run_path}: no {MANIFEST_NAME}, but durable failure "
            f"evidence is present -- {len(report.failed)} {FAILED_DIR_NAME} marker(s), "
            f"{len(report.partial)} {PARTIAL_SUFFIX} dir(s), "
            f"{len(report.unreadable)} unreadable h5(s). Episodes are known lost."
        )
        if not allow_incomplete:
            raise IncompleteEvalError(
                message + "\nRefusing to emit a success rate. Pass --allow-incomplete "
                "to override."
            )
        print(message, file=sys.stderr)

    if report.expected is None:
        print(
            "=" * 72 + "\n"
            f"WARNING: no {MANIFEST_NAME} under {run_path} and no --expected-episodes.\n"
            "         The success rate below is computed over the files that happen to\n"
            "         exist, NOT over the benchmark. It cannot be trusted as a board "
            "number.\n" + "=" * 72,
            file=sys.stderr,
        )
        return
    if report.complete:
        return

    message = report.describe(run_path)
    if not allow_incomplete:
        raise IncompleteEvalError(
            message + "\nRefusing to emit a success rate over a biased subset. "
            "Pass --allow-incomplete to override."
        )
    print(message, file=sys.stderr)


def eval_to_csv(
    run_path: str,
    policy_name: str,
    success_condition: str = "at-end",
    output_csv: str = "eval_results.csv",
    dt: float = 0.1,
    max_steps: int | None = None,
    completeness: CompletenessPolicy = CompletenessPolicy(),
):
    report_both = success_condition == "both"
    allow_incomplete = completeness.allow_incomplete

    report = _survey_episode_set(run_path, completeness.expected_episodes)
    combined_h5, report.unreadable = _combine_trajectories(
        run_path, allow_incomplete=allow_incomplete
    )
    # try/finally, not bare unlinks: every refusal path below raises, and so can
    # _episode_joint_jerk on a corrupt qpos row. The temp file must go regardless.
    try:
        _enforce_completeness(report, run_path, allow_incomplete=allow_incomplete)
        opts = ScoringOptions(
            success_condition=success_condition,
            dt=dt,
            max_steps=max_steps,
            allow_incomplete=allow_incomplete,
        )
        tally = _score_combined(combined_h5, opts)
        # Re-check with the SCORED count known. The pre-scoring pass can only see the
        # filesystem; only now can "every expected episode actually contributed a
        # scored trajectory" be enforced.
        report.scored = tally.total_n
        _enforce_completeness(report, run_path, allow_incomplete=allow_incomplete)

        rows = []
        for obj in sorted(tally.per_obj):
            d = tally.per_obj[obj]
            row, _ = _build_row(policy_name, obj, d["success"], d["total"],
                                d["jerk_joint"], report_both, d["oracle_success"])
            rows.append(row)

        if tally.total_n == 0 and not allow_incomplete:
            raise IncompleteEvalError(f"scored 0 episodes under {run_path}")

        overall_row, rate = _build_row(policy_name, "OVERALL", tally.total_s, tally.total_n,
                                       tally.all_jerk_joint, report_both, tally.total_os)
        rows.append(overall_row)

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # The `complete` COLUMN, not only the `#` provenance lines, is the load-bearing
        # half: any concat of these CSVs into a board sheet drops comment lines.
        for row in rows:
            row["complete"] = report.complete

        df = pd.DataFrame(rows)
        with open(output_csv, "w") as fout:
            fout.write(f"# policy_name: {policy_name}\n")
            fout.write(f"# run_path: {run_path}\n")
            fout.write(f"# dt: {dt}\n")
            fout.write(f"# max_steps: {max_steps}\n")
            fout.write(f"# expected_episodes: {report.expected}\n")
            fout.write(f"# scored_episodes: {tally.total_n}\n")
            if report.complete is not True:
                fout.write(
                    f"# INCOMPLETE: {len(report.missing)} missing {report.missing[:20]} "
                    f"failed_markers={len(report.failed)} partial_dirs={len(report.partial)} "
                    f"unreadable={len(report.unreadable)}\n"
                )
            df.to_csv(fout, index=False)

        summary = f"SR: {round(rate, 2)}%"
        if report_both:
            o_rate = 100.0 * tally.total_os / tally.total_n if tally.total_n else 0.0
            summary = f"at-end: {round(rate, 2)}% | oracle: {round(o_rate, 2)}%"
        print(f"\nSaved → {os.path.abspath(output_csv)} {summary} of {tally.total_n} episodes")

        return df
    finally:
        os.unlink(combined_h5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy and save results to CSV")
    parser.add_argument("run_path", help="Path to evaluation output directory")
    parser.add_argument("policy_name", help="Name of the policy")
    parser.add_argument("--success-condition", type=str, default="at-end", help="at-end | oracle | both")
    parser.add_argument("--output-csv", default="eval_results.csv", help="Output CSV file (default: eval_results.csv)")
    parser.add_argument("--dt", type=float, default=67/1000, help="Time step [s] (default: 0.1)")
    parser.add_argument("--steps-per-episode", type=int, default=None, help="Max steps per episode (default: None)")
    parser.add_argument("--expected-episodes", type=int, default=None,
                        help=f"Benchmark episode count; overrides {MANIFEST_NAME}")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Score a partial set anyway; stamps # INCOMPLETE and "
                             "complete=False into the CSV")

    args = parser.parse_args()

    eval_to_csv(
        run_path=args.run_path,
        policy_name=args.policy_name,
        #reward_threshold=args.reward_threshold,
        success_condition=args.success_condition,
        output_csv=args.output_csv,
        dt=args.dt,
        max_steps=args.steps_per_episode,
        completeness=CompletenessPolicy(
            expected_episodes=args.expected_episodes,
            allow_incomplete=args.allow_incomplete,
        ),
    )
