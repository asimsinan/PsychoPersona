"""Metadata-only profile prediction for the deployable cold-start evaluation (A1).

The oracle evaluation assigns a held-out respondent to a training profile using that
respondent's own scale means, which are derived from the answers the simulation is
supposed to produce. A deployable cold-start setting cannot do that. It must predict
profile membership from information available *before* the questionnaire is answered.

This script fits that predictor on the training split alone, using only the pre-response
context fields that already reach the prompt, and reports held-out accuracy against the
mixture-model profile label. That label is what selects the persona card and the profile
statistics for every profile-conditioned arm.

Candidate predictors are compared against the majority-profile rate *in the mixture
label space*, which is the relevant floor here and is not the same as the majority
baseline reported for profile recovery. Profile recovery scores in a nearest-centroid
space rebuilt from held-out scale means, so the two baselines differ whenever the
mixture and the recovery metric disagree on a label.

The accuracy reported here caps how well a cold-start arm can recover *profile
membership*. It does not cap respondent-level error, because an incorrect profile still
yields a coherent persona card, so item-level accuracy has to be measured by re-running
the arms.

Runs are supplied on the command line rather than pinned in the source, because profiles
are refitted whenever the split changes and a pinned run goes stale silently. The
production predictor used by a generation run lives in ``saykopersona.coldstart`` and is
fitted inside the run from that run's own training split, so it cannot drift from the
mixture it is predicting.

Usage
-----
    PYTHONPATH=src python3 scripts/coldstart_profile_predictor.py \
        --tr-run <run id> --bd-run <run id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import saykopersona.dataset_catalog as catalog
from saykopersona.coldstart import (context_columns, detect_multi_select,
                                    expand_multi_select, fit_coldstart_predictor,
                                    predict_coldstart_profiles)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs" / "experiment"


def _dataset_settings(name: str):
    if name == "anket_tr":
        return catalog.default_dataset_settings()
    if name == "bangladesh_sns":
        return catalog.bangladesh_sns_dataset_settings()
    raise SystemExit(f"Unknown dataset '{name}'.")


def _load(run_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "data" / f"{split}_split.csv")
    assignments = pd.read_csv(run_dir / "profiles" / f"{split}_profile_assignments.csv")
    keep = ["respondent_id", "profile_id_oracle"] if \
        "profile_id_oracle" in assignments.columns else ["respondent_id", "profile_id"]
    merged = frame.merge(assignments[keep], on="respondent_id")
    if "profile_id_oracle" in merged.columns:
        merged = merged.rename(columns={"profile_id_oracle": "profile_id"})
    return merged


def _candidate(categorical: list[str], numeric: list[str], kind: str, seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    if kind == "majority":
        model = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif kind == "forest_balanced":
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced",
            random_state=seed,
        )
    elif kind == "forest_plain":
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=seed,
        )
    else:
        raise SystemExit(f"unknown candidate '{kind}'")
    return Pipeline([("pre", pre), ("clf", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tr-run", required=True)
    parser.add_argument("--bd-run", required=True)
    # The production predictor is seeded with the split's random_state, so the analysis
    # must use the same value or the accuracy reported here will not be the accuracy the
    # generation runs actually achieve.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=Path("results/analysis/coldstart_profile_prediction.csv")
    )
    parser.add_argument(
        "--assignments-out", type=Path, default=Path("results/analysis/coldstart_assignments")
    )
    args = parser.parse_args()

    runs = {"anket_tr": args.tr_run, "bangladesh_sns": args.bd_run}
    rows: list[dict] = []
    args.assignments_out.mkdir(parents=True, exist_ok=True)

    for dataset, run in runs.items():
        settings = _dataset_settings(dataset)
        categorical, numeric = context_columns(settings)
        run_dir = RUNS_ROOT / run
        train = _load(run_dir, "train")
        # Apply the same multi-select expansion the deployed predictor uses. The Turkish
        # platforms field holds a comma-separated list, and without expanding it this
        # candidate table scores a different feature space from the one the runs used,
        # which silently produced a different accuracy for the selected model.
        multi = detect_multi_select(train, categorical)
        train, expanded = expand_multi_select(train, multi)
        categorical = [c for c in categorical if c not in multi]
        numeric = [*numeric, *expanded]
        y_train = train["profile_id"].astype(int)
        p_counts = y_train.value_counts().sort_index()

        for kind in ("majority", "logreg", "forest_balanced", "forest_plain"):
            model = _candidate(categorical, numeric, kind, args.seed)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            cv_acc = cross_val_score(model, train, y_train, cv=cv, scoring="accuracy")
            model.fit(train, y_train)
            for split in ("validation", "final_test"):
                held = _load(run_dir, split)
                if multi:
                    held, _ = expand_multi_select(held, multi)
                    for column in numeric:
                        if column not in held.columns:
                            held[column] = 0
                y_true = held["profile_id"].astype(int)
                y_pred = model.predict(held)
                # Floor for this task: always guess the largest *training* profile,
                # scored in the mixture label space the predictor targets.
                floor = float((y_true == p_counts.idxmax()).mean())
                rows.append({
                    "dataset": dataset, "split": split, "model": kind,
                    "n_train": len(train), "n": len(held),
                    "train_cv_accuracy": round(float(cv_acc.mean()), 4),
                    "train_cv_sd": round(float(cv_acc.std()), 4),
                    "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                    "majority_floor": round(floor, 4),
                    "lift_over_floor": round(float(accuracy_score(y_true, y_pred)) - floor, 4),
                    "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
                    "n_distinct_predicted": int(pd.Series(y_pred).nunique()),
                    "n_profiles": int(p_counts.size),
                    "run": run,
                })
        # Assignments are written by the same predictor a generation run uses, so the
        # A2 baseline recomputation and the C1 runs cannot disagree about which profile
        # a cold-start respondent was routed to. The candidate table above is reported
        # for the paper; this is the one that is actually deployed.
        production = fit_coldstart_predictor(train, dataset=settings, random_state=args.seed)
        for split in ("validation", "final_test"):
            held = _load(run_dir, split)
            out = held[["respondent_id"]].copy()
            out["profile_id_oracle"] = held["profile_id"].astype(int).to_numpy()
            out["profile_id_coldstart"] = predict_coldstart_profiles(production, held).to_numpy()
            out.to_csv(args.assignments_out / f"{dataset}_{split}_coldstart.csv", index=False)
        print(f"  {dataset}: production predictor selected '{production.selected}' "
              f"by train CV {production.cv_scores}")

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("=== A1: metadata-only profile prediction (target = mixture profile label) ===")
    cols = ["dataset", "split", "model", "accuracy", "majority_floor", "lift_over_floor",
            "macro_f1", "n_distinct_predicted", "n_profiles"]
    print(frame[cols].to_string(index=False))
    print(f"\nWrote {args.out}")
    print(f"Wrote cold-start assignments to {args.assignments_out}/")


if __name__ == "__main__":
    main()
