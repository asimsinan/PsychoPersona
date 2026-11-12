"""Oracle vs cold-start profile assignment for the profile-dependent baselines.

The two baselines that consume a profile label -- ``profile_center`` and
``within_profile`` -- can be evaluated twice: once with the oracle (response-derived)
assignment the manuscript currently uses, and once with the metadata-only prediction
produced by ``scripts/coldstart_profile_predictor.py``. The difference is the value of
knowing true profile membership, measured without generating anything.

This bounds the oracle advantage for the profile-conditioned LLM arms before those arms
are re-run. It is a bound, not a substitute: an incorrect profile label still yields a
coherent persona card for an LLM arm, so respondent-level error must still be measured
by running the arms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_nonllm_baselines import (  # noqa: E402
    _columns,
    _dataset_settings,
    _evaluate,
    _impute_train_items,
    _load_run,
    _recompute_scale_means,
    baseline_profile_center,
    baseline_within_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Runs supplied on the command line rather than pinned here. Profiles are refitted
# whenever the split changes, so a pinned run goes stale silently: the Turkish rows in
# an earlier version of this table were computed against a profile layer that no longer
# exists, and reported a cold-start assignment accuracy of 1.000 because that layer's
# label was a deterministic function of the usage band.
DEFAULT_RUNS = {
    "anket_tr": "20260908T071616227134Z-250d36e4",
    "bangladesh_sns": "20260908T071750014467Z-027e3193",
}
REPLICATES = 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path,
                        default=Path("results/analysis/coldstart_assignments"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/coldstart_baselines.csv"))
    parser.add_argument("--tr-run", default=DEFAULT_RUNS["anket_tr"])
    parser.add_argument("--bd-run", default=DEFAULT_RUNS["bangladesh_sns"])
    args = parser.parse_args()
    RUNS = {"anket_tr": args.tr_run, "bangladesh_sns": args.bd_run}

    rows = []
    for dataset, run in RUNS.items():
        settings = _dataset_settings(dataset)
        cols = _columns(settings)
        run_dir = REPO_ROOT / "results" / "runs" / "experiment" / run

        human, train = _load_run(run_dir, cols)
        human = _recompute_scale_means(human, cols)
        train = _recompute_scale_means(_impute_train_items(train, cols), cols)

        cold = pd.read_csv(args.assignments / f"{dataset}_final_test_coldstart.csv")
        agreement = float((cold["profile_id_oracle"] == cold["profile_id_coldstart"]).mean())

        for setting in ("oracle", "coldstart"):
            frame = human.copy()
            if setting == "coldstart":
                frame = frame.drop(columns=["profile_id"]).merge(
                    cold[["respondent_id", "profile_id_coldstart"]].rename(
                        columns={"profile_id_coldstart": "profile_id"}
                    ),
                    on="respondent_id",
                )
            pc = baseline_profile_center(frame, train, cols)
            rows.append({"dataset": dataset, "setting": setting, "assignment_accuracy": round(agreement, 4),
                         "method": "profile_center", **_evaluate(frame, pc, cols)})
            wp = [
                _evaluate(frame, baseline_within_profile(frame, train, cols, np.random.default_rng(20260907 + s)), cols)
                for s in range(REPLICATES)
            ]
            mean = {k: sum(d[k] for d in wp) / len(wp) for k in wp[0]}
            rows.append({"dataset": dataset, "setting": setting, "assignment_accuracy": round(agreement, 4),
                         "method": f"within_profile (mean of {REPLICATES})", **mean})

    frame = pd.DataFrame(rows).round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("=== A2: oracle vs cold-start profile assignment, non-LLM baselines (final test) ===")
    print(frame.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
