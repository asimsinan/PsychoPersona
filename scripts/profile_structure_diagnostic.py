"""Diagnose what the discovered empirical profiles actually encode.

Profiles are fitted with a Gaussian mixture over three scale means plus the daily
usage-hour midpoint. The usage midpoint is a *discretised* variable derived from the
usage band, so it takes only a handful of distinct values. A full-covariance mixture
can drive the variance of such a dimension toward zero, which inflates the likelihood
without number and lets a single discrete feature determine the partition.

This script reports, per dataset: how much between-profile variance each fitting
feature explains, whether the profile label is a deterministic function of the usage
band, and the information criteria across candidate profile counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

import saykopersona.dataset_catalog as catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "anket_tr": "20260413T151353920083Z-e9b99971",
    "bangladesh_sns": "20260413T093157879555Z-712c7624",
}
FEATURES = {
    "anket_tr": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean",
                 "usage_hours_midpoint"],
    "bangladesh_sns": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean",
                       "usage_hours_midpoint"],
}


def _settings(name):
    return (catalog.default_dataset_settings() if name == "anket_tr"
            else catalog.bangladesh_sns_dataset_settings())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/profile_structure_diagnostic.csv"))
    args = parser.parse_args()

    rows = []
    for dataset, run in RUNS.items():
        run_dir = REPO_ROOT / "results" / "runs" / "experiment" / run
        settings = _settings(dataset)
        train = pd.read_csv(run_dir / "data" / "train_split.csv")
        pa = pd.read_csv(run_dir / "profiles" / "train_profile_assignments.csv")
        d = train.merge(pa[["respondent_id", "profile_id"]], on="respondent_id")
        for scale in settings.scales:
            d[scale.mean_column] = d[list(scale.item_columns)].mean(axis=1)

        groups = [g for _, g in d.groupby("profile_id")]
        for feature in FEATURES[dataset]:
            samples = [g[feature].dropna().to_numpy() for g in groups]
            f_stat, p = stats.f_oneway(*samples)
            grand = d[feature].mean()
            ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
            ss_total = ((d[feature] - grand) ** 2).sum()
            rows.append({
                "dataset": dataset,
                "feature": feature,
                "eta_squared": round(float(ss_between / ss_total), 4),
                "f_stat": round(float(f_stat), 1),
                "n_distinct_values": int(d[feature].nunique()),
            })

        ct = pd.crosstab(d["usage_band"], d["profile_id"])
        deterministic = bool(((ct > 0).sum(axis=1) <= 1).all())
        scores = pd.read_csv(run_dir / "profiles" / "profile_model_scores.csv")
        print("=" * 74)
        print(f"{dataset}: profile label determined by usage band alone = {deterministic}")
        print(f"  distinct usage midpoints per profile: "
              f"{d.groupby('profile_id')['usage_hours_midpoint'].nunique().to_dict()}")
        print("  model selection scores:")
        print(scores.to_string(index=False))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("\n=== Variance in each fitting feature explained by the profile label ===")
    print(frame.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
