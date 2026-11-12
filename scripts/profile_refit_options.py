"""Decision support: what the Turkish profile layer looks like under alternative feature sets.

The current Turkish mixture is fitted on three scale means plus the daily usage-hour
midpoint. The midpoint is discretised (four distinct values), which lets a
full-covariance mixture collapse that dimension's variance and partition on it alone.

This script refits the profile layer under three feature sets and reports, for each:
the selected profile count, whether the information criteria stay non-degenerate, the
profile sizes, how much of each feature's variance the partition explains, whether the
partition is still a usage-band relabeling, how predictable the label is from
pre-response metadata alone (the cold-start ceiling), and how stable the selected count
is across random splits.

No option is applied. This only shows what each would give.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from saykopersona.data import (  # noqa: E402
    build_stratified_three_way_split,
    load_raw_survey,
    score_survey_dataframe,
)
from saykopersona.dataset_catalog import default_dataset_settings  # noqa: E402
from saykopersona.profiles import fit_empirical_profiles, predict_profiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SCALES = ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"]

OPTIONS = {
    "current (scales + usage midpoint)": SCALES + ["usage_hours_midpoint"],
    "option A (scale means only)": SCALES,
}


def _coldstart_accuracy(train, test, settings) -> float:
    names = [m.name for m in settings.metadata_fields if getattr(m, "include_in_context", False)]
    numeric = [n for n in names if n.endswith("_midpoint")]
    categorical = [n for n in names if n not in numeric]
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
            ("num", StandardScaler(), numeric)])),
        ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                       class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(train, train["profile_id"].astype(int))
    return float(accuracy_score(test["profile_id"].astype(int), pipe.predict(test)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=25)
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/profile_refit_options.csv"))
    args = parser.parse_args()

    settings = default_dataset_settings()
    scored = score_survey_dataframe(
        load_raw_survey(REPO_ROOT / "anket.xlsx", dataset=settings), dataset=settings)
    train, _val, test = build_stratified_three_way_split(
        scored, validation_size=0.1, final_test_size=0.1, random_state=SEED, dataset=settings)

    rows = []
    for label, features in OPTIONS.items():
        fit_frame = train.dropna(subset=features)
        result = fit_empirical_profiles(
            fit_frame, feature_columns=features, random_state=SEED,
            anchor_columns=list(settings.profile_anchor_columns))
        assigned = result.assignments
        test_a = test.dropna(subset=features).copy()
        test_a["profile_id"] = predict_profiles(result, test_a).to_numpy()

        scores = result.model_scores
        chosen = scores[scores["selected"]].iloc[0]
        sizes = assigned["profile_id"].value_counts().sort_index()
        ct = pd.crosstab(assigned["usage_band"], assigned["profile_id"])
        usage_determined = bool(((ct > 0).sum(axis=1) <= 1).all())

        etas = {}
        for feature in SCALES + ["usage_hours_midpoint"]:
            groups = [g[feature].dropna().to_numpy() for _, g in assigned.groupby("profile_id")]
            grand = assigned[feature].mean()
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ss_t = ((assigned[feature] - grand) ** 2).sum()
            etas[feature] = round(float(ss_b / ss_t), 4)

        ks = []
        for seed in range(args.splits):
            tr2, _v2, _t2 = build_stratified_three_way_split(
                scored, validation_size=0.1, final_test_size=0.1, random_state=seed,
                dataset=settings)
            r2 = fit_empirical_profiles(tr2.dropna(subset=features), feature_columns=features,
                                        random_state=seed,
                                        anchor_columns=list(settings.profile_anchor_columns))
            ks.append(r2.selected_k)

        rows.append({
            "feature_set": label, "selected_k": result.selected_k,
            "bic_at_selected": round(float(chosen["bic"]), 1),
            "bic_negative_anywhere": bool((scores["bic"] < 0).any()),
            "silhouette": round(float(chosen["silhouette"]), 4),
            "profile_sizes": "/".join(str(int(v)) for v in sizes),
            "usage_determined": usage_determined,
            **{f"eta2_{k.replace('_mean','').replace('usage_hours_midpoint','usage')}": v
               for k, v in etas.items()},
            "coldstart_accuracy": round(_coldstart_accuracy(assigned, test_a, settings), 4),
            f"k_across_{args.splits}_splits": str(dict(pd.Series(ks).value_counts().sort_index())),
        })
        print(f"\n--- {label} ---")
        print(scores.to_string(index=False))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("\n=== Turkish profile layer under alternative feature sets ===")
    print(frame.T.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
