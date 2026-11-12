"""Split sensitivity of the benchmark metrics (Reviewer 1, comment 2).

The manuscript reports a single 80/10/10 split with ``random_state=42``. Reviewer 1
observes that correlation RMSE, profile recovery and uniqueness may be sensitive to that
particular split. This script repeats the whole non-LLM half of the pipeline -- split,
profile fitting, profile assignment, baseline generation, evaluation -- across many
random splits, and reports the between-split dispersion of every metric.

It also records, per split, the selected profile count and whether the discovered
profile partition is a deterministic function of the usage band, so that the stability
of the profile structure itself can be judged rather than assumed.

This covers the non-LLM reference points. The LLM arms are addressed separately by
re-running them across a subset of these splits.
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
    _recompute_scale_means,
    baseline_marginal,
    baseline_nn_metadata,
    baseline_profile_center,
    baseline_within_profile,
)

from saykopersona.data import (  # noqa: E402
    build_stratified_three_way_split,
    load_raw_survey,
    score_survey_dataframe,
)
from saykopersona.profiles import fit_empirical_profiles, predict_profiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "anket_tr": REPO_ROOT / "anket.xlsx",
    "bangladesh_sns": REPO_ROOT / "datasets" / "bangladesh_sns" / "mmc2.xlsx",
}
# Turkish fits on scale means only: the usage midpoint is discretised and a
# full-covariance mixture partitions on it alone (see config.ProfileSettings).
FEATURES = {
    "anket_tr": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"],
    # Scale means only, on both datasets. The usage midpoint is printed in the prompt
    # as respondent context, so fitting the mixture on it makes the profile partly
    # recoverable from the prompt; it is retained as a profile anchor instead.
    "bangladesh_sns": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/split_robustness.csv"))
    parser.add_argument("--summary-out", type=Path,
                        default=Path("results/analysis/split_robustness_summary.csv"))
    args = parser.parse_args()

    rows = []
    for dataset, path in PATHS.items():
        settings = _dataset_settings(dataset)
        cols = _columns(settings)
        scored = score_survey_dataframe(load_raw_survey(path, dataset=settings), dataset=settings)

        for seed in range(args.splits):
            train, _val, test = build_stratified_three_way_split(
                scored, validation_size=0.1, final_test_size=0.1, random_state=seed,
                dataset=settings,
            )
            fit_frame = train.dropna(subset=FEATURES[dataset])
            result = fit_empirical_profiles(
                fit_frame, feature_columns=FEATURES[dataset], random_state=seed,
                anchor_columns=list(settings.profile_anchor_columns),
            )
            train_p = result.assignments.copy()
            test_p = test.copy()
            test_p["profile_id"] = predict_profiles(result, test_p.dropna(
                subset=FEATURES[dataset])).reindex(test_p.index)
            test_p = test_p.dropna(subset=["profile_id"])
            test_p["profile_id"] = test_p["profile_id"].astype(int)

            train_p = _recompute_scale_means(_impute_train_items(train_p, cols), cols)
            test_p = _recompute_scale_means(test_p, cols)

            crosstab = pd.crosstab(train_p["usage_band"], train_p["profile_id"])
            usage_determined = bool(((crosstab > 0).sum(axis=1) <= 1).all())
            rng = np.random.default_rng(20260907 + seed)

            for name, sim in [
                ("profile_center", baseline_profile_center(test_p, train_p, cols)),
                ("within_profile", baseline_within_profile(test_p, train_p, cols, rng)),
                ("nn_metadata", baseline_nn_metadata(test_p, train_p, cols)),
                ("marginal", baseline_marginal(test_p, train_p, cols, rng)),
            ]:
                rows.append({
                    "dataset": dataset, "seed": seed, "method": name,
                    "selected_k": result.selected_k,
                    "profile_usage_determined": usage_determined,
                    "n_test": len(test_p), **_evaluate(test_p, sim, cols),
                })

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    metrics = ["CorrRMSE", "WithinRMSE", "PRA", "RespScaleRMSE", "RespItemMAE", "ItemEMR",
               "SimUniqueRatio"]
    summary = (frame.groupby(["dataset", "method"])[metrics]
               .agg(["mean", "std", "min", "max"]).round(4))
    summary.to_csv(args.summary_out)

    print(f"=== A3: split robustness across {args.splits} random splits ===")
    for dataset in PATHS:
        sub = frame[frame["dataset"] == dataset]
        ks = sub.groupby("seed")["selected_k"].first()
        det = sub.groupby("seed")["profile_usage_determined"].first()
        print(f"\n{dataset}: selected profile count P across splits -> {dict(ks.value_counts().sort_index())}")
        print(f"  partition is a deterministic function of usage band in "
              f"{int(det.sum())}/{len(det)} splits")
    print("\n--- between-split dispersion (mean [min, max], SD) ---")
    for dataset in PATHS:
        print(f"\n{dataset}")
        for method in ["profile_center", "within_profile", "marginal"]:
            sub = frame[(frame["dataset"] == dataset) & (frame["method"] == method)]
            bits = [f"{m} {sub[m].mean():.3f} [{sub[m].min():.3f},{sub[m].max():.3f}] sd={sub[m].std():.3f}"
                    for m in ["CorrRMSE", "PRA", "SimUniqueRatio"]]
            print(f"  {method:15s} " + " | ".join(bits))
    print(f"\nWrote {args.out}\nWrote {args.summary_out}")


if __name__ == "__main__":
    main()
