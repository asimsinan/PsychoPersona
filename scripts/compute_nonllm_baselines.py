"""Compute non-LLM baseline analyses for the PsychoPersona BRM benchmark.

This script implements simple, transparent baselines that reuse the *same*
held-out profile assignment and the *same* evaluation pipeline
(``saykopersona.evaluation.evaluate_simulation``) as the LLM arms, so that
reviewers can judge whether LLM generation adds value beyond profile
statistics and training-set resampling.

Baselines
---------
1. ``profile_center``      - deterministic: each held-out respondent receives the
                             training-set, per-profile mean item vector (rounded
                             and clipped to the observed item range).
2. ``within_profile``      - stochastic: sample one training respondent from the
                             same assigned profile and copy their item responses.
3. ``nn_metadata``         - deterministic: nearest training respondent by
                             pre-response metadata (one-hot categoricals + scaled
                             numeric), copy that donor's item responses.
4. ``marginal``            - stochastic: each item is drawn independently from the
                             training-set marginal distribution of that item.

For the two stochastic baselines we run ``--replicates`` seeded replications and
report mean +/- SD across replications.

Usage
-----
    PYTHONPATH=src python3 scripts/compute_nonllm_baselines.py \
        --run-dir results/runs/experiment/<run-id> \
        --dataset anket_tr \
        --replicates 50 \
        --out results/analysis/nonllm_baselines_anket_tr_sadp.csv
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import saykopersona.dataset_catalog as catalog
from saykopersona.evaluation import evaluate_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Dataset wiring                                                              #
# --------------------------------------------------------------------------- #
def _dataset_settings(name: str):
    if name == "anket_tr":
        return catalog.default_dataset_settings()
    if name == "bangladesh_sns":
        return catalog.bangladesh_sns_dataset_settings()
    raise SystemExit(f"Unknown dataset '{name}'. Use anket_tr or bangladesh_sns.")


@dataclass(slots=True)
class DatasetColumns:
    scale_columns: list[str]
    item_columns: list[str]
    metadata_columns: list[str]
    scale_to_items: dict[str, list[str]]


def _columns(settings) -> DatasetColumns:
    scale_columns = [s.mean_column for s in settings.scales]
    item_columns = [item for s in settings.scales for item in s.item_columns]
    scale_to_items = {s.mean_column: list(s.item_columns) for s in settings.scales}
    metadata_columns = [
        m.name for m in settings.metadata_fields if getattr(m, "include_in_context", False)
    ]
    return DatasetColumns(
        scale_columns=scale_columns,
        item_columns=item_columns,
        metadata_columns=metadata_columns,
        scale_to_items=scale_to_items,
    )


def _recompute_scale_means(frame: pd.DataFrame, cols: DatasetColumns) -> pd.DataFrame:
    out = frame.copy()
    for scale, items in cols.scale_to_items.items():
        out[scale] = out[items].astype(float).mean(axis=1)
    return out


def _impute_train_items(train: pd.DataFrame, cols: DatasetColumns) -> pd.DataFrame:
    """Replace missing training item responses with the rounded column median.

    This keeps the donor-based baselines (``within_profile``, ``nn_metadata``)
    well-defined when an individual training respondent skipped an item.
    """
    out = train.copy()
    for item in cols.item_columns:
        series = out[item].astype(float)
        if series.isna().any():
            median = series.median()
            out[item] = series.fillna(round(median) if pd.notna(median) else 0.0)
    return out


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def _load_run(run_dir: Path, cols: DatasetColumns):
    """Return (human_final_test, train) frames each with a profile_id column."""
    final_test = pd.read_csv(run_dir / "data" / "final_test_split.csv")
    train = pd.read_csv(run_dir / "data" / "train_split.csv")
    final_pa = pd.read_csv(run_dir / "profiles" / "final_test_profile_assignments.csv")
    train_pa = pd.read_csv(run_dir / "profiles" / "train_profile_assignments.csv")

    final_test = final_test.merge(
        final_pa[["respondent_id", "profile_id"]], on="respondent_id", how="left"
    )
    train = train.merge(
        train_pa[["respondent_id", "profile_id"]], on="respondent_id", how="left"
    )
    return final_test, train


# --------------------------------------------------------------------------- #
# Baseline generators                                                         #
# --------------------------------------------------------------------------- #
def _item_bounds(train: pd.DataFrame, cols: DatasetColumns) -> dict[str, tuple[int, int]]:
    bounds: dict[str, tuple[int, int]] = {}
    for item in cols.item_columns:
        series = train[item].dropna().astype(float)
        bounds[item] = (int(round(series.min())), int(round(series.max())))
    return bounds


def baseline_profile_center(
    human: pd.DataFrame, train: pd.DataFrame, cols: DatasetColumns
) -> pd.DataFrame:
    bounds = _item_bounds(train, cols)
    centers = train.groupby("profile_id")[cols.item_columns].mean()
    global_center = train[cols.item_columns].mean()
    rows = []
    for _, r in human.iterrows():
        pid = r["profile_id"]
        center = centers.loc[pid] if pid in centers.index else global_center
        row = {"respondent_id": r["respondent_id"], "profile_id": pid}
        for item in cols.item_columns:
            lo, hi = bounds[item]
            row[item] = int(np.clip(round(float(center[item])), lo, hi))
        rows.append(row)
    sim = pd.DataFrame(rows)
    return _recompute_scale_means(sim, cols)


def baseline_within_profile(
    human: pd.DataFrame, train: pd.DataFrame, cols: DatasetColumns, rng: np.random.Generator
) -> pd.DataFrame:
    by_profile = {pid: g for pid, g in train.groupby("profile_id")}
    rows = []
    for _, r in human.iterrows():
        pid = r["profile_id"]
        pool = by_profile.get(pid)
        if pool is None or pool.empty:
            pool = train
        donor = pool.iloc[int(rng.integers(len(pool)))]
        row = {"respondent_id": r["respondent_id"], "profile_id": pid}
        for item in cols.item_columns:
            row[item] = float(donor[item])
        rows.append(row)
    sim = pd.DataFrame(rows)
    return _recompute_scale_means(sim, cols)


def baseline_marginal(
    human: pd.DataFrame, train: pd.DataFrame, cols: DatasetColumns, rng: np.random.Generator
) -> pd.DataFrame:
    pools = {item: train[item].dropna().astype(float).to_numpy() for item in cols.item_columns}
    rows = []
    for _, r in human.iterrows():
        row = {"respondent_id": r["respondent_id"], "profile_id": r["profile_id"]}
        for item in cols.item_columns:
            pool = pools[item]
            row[item] = float(pool[int(rng.integers(len(pool)))])
        rows.append(row)
    sim = pd.DataFrame(rows)
    return _recompute_scale_means(sim, cols)


def _encode_metadata(
    human: pd.DataFrame, train: pd.DataFrame, cols: DatasetColumns
) -> tuple[np.ndarray, np.ndarray]:
    usable = [c for c in cols.metadata_columns if c in train.columns and c in human.columns]
    # Drop free-text/timestamp columns that are not useful for matching.
    drop = {"timestamp", "platforms", "primary_platform"}
    usable = [c for c in usable if c not in drop]
    combined = pd.concat([train[usable], human[usable]], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined.astype(str), dummy_na=True)
    # Standardise numeric usage-hours separately if present.
    matrix = encoded.to_numpy(dtype=float)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / std
    n_train = len(train)
    return matrix[:n_train], matrix[n_train:]


def baseline_nn_metadata(
    human: pd.DataFrame, train: pd.DataFrame, cols: DatasetColumns
) -> pd.DataFrame:
    train_x, human_x = _encode_metadata(human, train, cols)
    train_items = train[cols.item_columns].astype(float).to_numpy()
    rows = []
    for i, (_, r) in enumerate(human.iterrows()):
        dist = np.sum((train_x - human_x[i][None, :]) ** 2, axis=1)
        donor_idx = int(np.argmin(dist))
        row = {"respondent_id": r["respondent_id"], "profile_id": r["profile_id"]}
        for j, item in enumerate(cols.item_columns):
            row[item] = float(train_items[donor_idx, j])
        rows.append(row)
    sim = pd.DataFrame(rows)
    return _recompute_scale_means(sim, cols)


# --------------------------------------------------------------------------- #
# Evaluation harness                                                          #
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    "CorrRMSE",
    "WithinRMSE",
    "PRA",
    "RespScaleRMSE",
    "RespItemMAE",
    "ItemEMR",
    "FullVectorEMR",
    "SimUniqueRatio",
]


def _report_to_row(rep) -> dict[str, float]:
    ind = rep.individual_level_metrics
    div = rep.response_pattern_diversity
    return {
        "CorrRMSE": rep.correlation_rmse,
        "WithinRMSE": rep.within_profile_scale_rmse,
        "PRA": rep.profile_recovery_accuracy,
        "RespScaleRMSE": ind.get("respondent_scale_rmse_mean", float("nan")),
        "RespItemMAE": ind.get("respondent_item_mae_mean", float("nan")),
        "ItemEMR": ind.get("respondent_item_exact_match_mean", float("nan")),
        "FullVectorEMR": ind.get("respondent_item_all_match_ratio", float("nan")),
        "SimUniqueRatio": div.get("simulated_unique_ratio", float("nan")),
    }


def _evaluate(human: pd.DataFrame, sim: pd.DataFrame, cols: DatasetColumns) -> dict[str, float]:
    rep = evaluate_simulation(
        human,
        sim,
        scale_columns=cols.scale_columns,
        item_columns=cols.item_columns,
        profile_column="profile_id",
    )
    return _report_to_row(rep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["anket_tr", "bangladesh_sns"])
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    settings = _dataset_settings(args.dataset)
    cols = _columns(settings)

    human, train = _load_run(run_dir, cols)
    train = _impute_train_items(train, cols)
    human = _recompute_scale_means(human, cols)
    train = _recompute_scale_means(train, cols)

    results: list[dict] = []

    # LLM arm (same run, identical pipeline) -------------------------------- #
    sim_path = run_dir / "simulation" / "simulated_responses.csv"
    if sim_path.exists():
        llm_sim = pd.read_csv(sim_path)
        llm_sim = _recompute_scale_means(llm_sim, cols)
        row = _evaluate(human, llm_sim, cols)
        row.update({"method": "LLM (this run)", "kind": "llm", "replicates": 1})
        results.append(row)

    # Deterministic baselines ---------------------------------------------- #
    for name, fn in [
        ("profile_center", baseline_profile_center),
        ("nn_metadata", baseline_nn_metadata),
    ]:
        sim = fn(human, train, cols)
        row = _evaluate(human, sim, cols)
        row.update({"method": name, "kind": "deterministic", "replicates": 1})
        results.append(row)

    # Stochastic baselines (mean +/- SD over replicates) ------------------- #
    for name, fn in [
        ("within_profile", baseline_within_profile),
        ("marginal", baseline_marginal),
    ]:
        per_seed: list[dict[str, float]] = []
        for seed in range(args.replicates):
            rng = np.random.default_rng(seed)
            sim = fn(human, train, cols, rng)
            per_seed.append(_evaluate(human, sim, cols))
        agg: dict[str, float] = {}
        for key in METRIC_KEYS:
            vals = [d[key] for d in per_seed if not np.isnan(d[key])]
            agg[key] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{key}_sd"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        agg.update({"method": name, "kind": "stochastic", "replicates": args.replicates})
        results.append(agg)

    df = pd.DataFrame(results)
    ordered = ["method", "kind", "replicates", *METRIC_KEYS]
    sd_cols = [f"{k}_sd" for k in METRIC_KEYS if f"{k}_sd" in df.columns]
    df = df[[c for c in ordered if c in df.columns] + sd_cols]

    print(f"\n=== Non-LLM baselines | dataset={args.dataset} | n_final_test={len(human)} ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.round(4).to_string(index=False))

    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.round(6).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
