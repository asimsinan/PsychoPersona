"""Why metadata appears to predict profile membership on one dataset and not the other.

The cold-start setting asks whether a held-out respondent can be routed to a training
profile using only pre-response context. Measured as shipped, the answer looks different
across the two datasets. Metadata predicts the Bangladesh profile label well above the
majority floor and the Turkish label not at all. This script establishes that the
difference is not a property of the populations but of the two profile layers.

Three checks, in order of decisiveness.

1. Can metadata predict the *scale means* at all? This is independent of any clustering.
   Cross-validated R^2 is at or below zero for every scale on both datasets, so context
   fields carry no linear or tree-recoverable signal about the psychometric constructs.

2. Where does the Bangladesh lift come from? The Bangladesh mixture is fitted on three
   scale means *plus* ``usage_hours_midpoint``, which is a context field the prompt
   already shows. Predicting the profile from that one feature alone matches or beats
   using every context field together, and dropping it puts accuracy below the majority
   floor.

3. What happens if the Bangladesh layer is refitted without it? The profile count moves
   from four to six and the cold-start lift falls to exactly zero, with the majority
   classifier selected. Bangladesh then behaves as Turkish does.

The Turkish layer already excludes this feature, because the C0 diagnostic found the
Turkish label was a deterministic relabeling of the usage band at eta^2 = 1.000. The
Bangladesh feature sits at eta^2 = 0.494, so it contributes without dominating and the
"do all features contribute" form of the diagnostic passes. That form of the question is
insufficient. What matters for cold-start is whether *any* fitting feature is also
prompt-visible metadata, because the predictor can then recover part of its own input.

Usage
-----
    PYTHONPATH=src python3 scripts/coldstart_leakage_diagnostic.py \
        --tr-run <run id> --bd-run <run id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import saykopersona.dataset_catalog as catalog
from saykopersona.coldstart import context_columns, fit_coldstart_predictor, \
    predict_coldstart_profiles
from saykopersona.profiles import fit_empirical_profiles, predict_profiles

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs" / "experiment"
SEED = 20260907

FEATURES = {
    "anket_tr": {
        "shipped": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"],
        "psychological_only": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"],
    },
    "bangladesh_sns": {
        "shipped": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean", "usage_hours_midpoint"],
        "psychological_only": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean"],
    },
}


def _settings(name: str):
    if name == "anket_tr":
        return catalog.default_dataset_settings()
    if name == "bangladesh_sns":
        return catalog.bangladesh_sns_dataset_settings()
    raise SystemExit(f"Unknown dataset '{name}'.")


def _pipe(categorical, numeric, clf):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _eta2(frame: pd.DataFrame, feature: str, label: str = "profile_id") -> float:
    x = frame[feature].astype(float)
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = sum(len(g) * (g[feature].mean() - grand) ** 2 for _, g in frame.groupby(label))
    return float(ss_between / ss_total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tr-run", required=True)
    parser.add_argument("--bd-run", required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/coldstart_leakage_diagnostic.json"))
    args = parser.parse_args()

    report: dict[str, object] = {}
    runs = {"anket_tr": args.tr_run, "bangladesh_sns": args.bd_run}

    print("=" * 78)
    print("1. Can metadata predict the scale means at all?  (5-fold CV R^2)")
    print("   R^2 <= 0 means no better than predicting the sample mean.")
    print("=" * 78)
    r2_report = {}
    for dataset, run in runs.items():
        settings = _settings(dataset)
        categorical, numeric = context_columns(settings)
        # 'platforms' is a comma-separated multi-select with 91 combination levels on
        # 537 Turkish rows. One-hot encoding the combination is the wrong treatment and
        # min_frequency collapses most of it, so it is excluded here; multi-hot encoding
        # it changes accuracy by under 0.002, reported in check 2.
        categorical = [c for c in categorical if c != "platforms"]
        frame = pd.read_csv(RUNS_ROOT / run / "data" / "train_split.csv")
        print(f"\n  {dataset}")
        r2_report[dataset] = {}
        for scale in settings.scales:
            col = scale.mean_column
            target = frame[col].astype(float)
            keep = target.notna()
            score = cross_val_score(
                _pipe(categorical, numeric,
                      RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=1)),
                frame[keep], target[keep],
                cv=KFold(5, shuffle=True, random_state=1), scoring="r2",
            ).mean()
            r2_report[dataset][col] = round(float(score), 4)
            print(f"    {col:28s} CV R^2 = {score:+.4f}")
    report["metadata_predicts_scale_means_cv_r2"] = r2_report

    print("\n" + "=" * 78)
    print("2. Where does the Bangladesh lift come from?")
    print("=" * 78)
    lift = {}
    for dataset, run in runs.items():
        settings = _settings(dataset)
        categorical, numeric = context_columns(settings)
        categorical = [c for c in categorical if c != "platforms"]
        frame = pd.read_csv(RUNS_ROOT / run / "data" / "train_split.csv").merge(
            pd.read_csv(RUNS_ROOT / run / "profiles" / "train_profile_assignments.csv")
            [["respondent_id", "profile_id"]], on="respondent_id")
        target = frame["profile_id"].astype(int)
        floor = float(target.value_counts().max() / len(target))
        forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED)
        sets = {
            "all context fields": (categorical, numeric),
            "usage only": ([], ["usage_hours_midpoint"]),
            "all except usage": ([c for c in categorical if c != "usage_band"],
                                 [n for n in numeric if n != "usage_hours_midpoint"]),
        }
        print(f"\n  {dataset}   majority floor = {floor:.4f}")
        lift[dataset] = {"majority_floor": round(floor, 4)}
        for tag, (cc, nn) in sets.items():
            if not cc and not nn:
                continue
            acc = cross_val_score(
                _pipe(cc, nn, forest), frame, target,
                cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy",
            ).mean()
            lift[dataset][tag] = round(float(acc), 4)
            print(f"    {tag:22s} CV accuracy = {acc:.4f}   lift = {acc - floor:+.4f}")
        print("    eta^2 of each fitting feature against the profile label:")
        for feat in FEATURES[dataset]["shipped"]:
            flag = "  <-- prompt-visible metadata" if feat == "usage_hours_midpoint" else ""
            value = _eta2(frame, feat)
            lift[dataset][f"eta2_{feat}"] = round(value, 4)
            print(f"      {feat:26s} {value:.3f}{flag}")
    report["lift_decomposition"] = lift

    print("\n" + "=" * 78)
    print("3. Refitting each layer on the psychological scales alone")
    print("=" * 78)
    refit = {}
    for dataset, run in runs.items():
        settings = _settings(dataset)
        train = pd.read_csv(RUNS_ROOT / run / "data" / "train_split.csv")
        test = pd.read_csv(RUNS_ROOT / run / "data" / "final_test_split.csv")
        refit[dataset] = {}
        print(f"\n  {dataset}")
        for tag in ("shipped", "psychological_only"):
            feats = FEATURES[dataset][tag]
            result = fit_empirical_profiles(
                train, feature_columns=feats, cluster_range=range(2, 7),
                random_state=42, anchor_columns=settings.profile_anchor_columns,
            )
            held = test.copy()
            held["profile_id"] = predict_profiles(result, held)
            predictor = fit_coldstart_predictor(result.assignments, dataset=settings,
                                                random_state=42)
            predicted = predict_coldstart_profiles(predictor, held)
            majority = result.assignments["profile_id"].value_counts().idxmax()
            floor = float((held["profile_id"] == majority).mean())
            acc = float((predicted.to_numpy() == held["profile_id"].to_numpy()).mean())
            refit[dataset][tag] = {
                "n_features": len(feats), "selected_P": int(result.selected_k),
                "coldstart_accuracy": round(acc, 4), "majority_floor": round(floor, 4),
                "lift": round(acc - floor, 4), "selected_predictor": predictor.selected,
            }
            print(f"    {tag:20s} P={result.selected_k}  accuracy={acc:.4f}  "
                  f"floor={floor:.4f}  lift={acc - floor:+.4f}  "
                  f"predictor={predictor.selected}")
    report["refit_without_metadata_feature"] = refit

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
