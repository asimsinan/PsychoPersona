"""Bootstrap confidence intervals and effect sizes for the PsychoPersona BRM benchmark.

For each (dataset, arm) we recompute *per-respondent* error metrics on the held-out
final-test split, then bootstrap over respondents to obtain 95% confidence intervals
for the metric means. We also compute paired effect sizes (standardized mean
difference, Cohen's d_z) and mean deltas with bootstrap CIs for the key method
contrasts: SADP vs BPP and SADP vs the profile-center baseline.

The per-respondent computations mirror ``saykopersona.evaluation`` exactly so the
bootstrap means reproduce the point estimates already reported in the manuscript.

Usage
-----
    PYTHONPATH=src python3 scripts/compute_uncertainty.py \
        --out results/analysis/uncertainty_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import saykopersona.dataset_catalog as catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 20260607

# Canonical run per (dataset, arm) for the final-test split, read from the run index so
# that this script, the tables and the manuscript cannot drift apart. Rebuild the index
# with scripts/build_run_index.py after adding runs.
def _load_runs() -> dict[tuple[str, str], str]:
    import pandas as _pd
    idx = _pd.read_csv(REPO_ROOT / "results" / "analysis" / "run_index.csv")
    idx = idx[(idx.split == "final_test") & (idx.model == "GPT-5.4")]
    return {(r.dataset, r.arm): r.run for r in idx.itertuples()}


RUNS: dict[tuple[str, str], str] = _load_runs()


@dataclass(slots=True)
class Cols:
    scale_columns: list[str]
    item_columns: list[str]
    scale_to_items: dict[str, list[str]]


def _dataset_settings(name: str):
    if name == "anket_tr":
        return catalog.default_dataset_settings()
    if name == "bangladesh_sns":
        return catalog.bangladesh_sns_dataset_settings()
    raise SystemExit(f"Unknown dataset '{name}'.")


def _columns(settings) -> Cols:
    scale_columns = [s.mean_column for s in settings.scales]
    item_columns = [item for s in settings.scales for item in s.item_columns]
    scale_to_items = {s.mean_column: list(s.item_columns) for s in settings.scales}
    return Cols(scale_columns, item_columns, scale_to_items)


def _recompute_scale_means(frame: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    out = frame.copy()
    for scale, items in cols.scale_to_items.items():
        out[scale] = out[items].astype(float).mean(axis=1)
    return out


def _impute_train_items(train: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    out = train.copy()
    for item in cols.item_columns:
        series = out[item].astype(float)
        if series.isna().any():
            median = series.median()
            out[item] = series.fillna(round(median) if pd.notna(median) else 0.0)
    return out


def _load_run(run_dir: Path, cols: Cols):
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


def _baseline_profile_center(human: pd.DataFrame, train: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    bounds = {}
    for item in cols.item_columns:
        s = train[item].dropna().astype(float)
        bounds[item] = (int(round(s.min())), int(round(s.max())))
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
    return _recompute_scale_means(pd.DataFrame(rows), cols)


def _per_respondent_metrics(
    human: pd.DataFrame, sim: pd.DataFrame, cols: Cols
) -> dict[str, np.ndarray]:
    """Return per-respondent metric vectors aligned on respondent_id."""
    joined = human[["respondent_id", *cols.scale_columns, *cols.item_columns]].merge(
        sim[["respondent_id", *cols.scale_columns, *cols.item_columns]],
        on="respondent_id",
        suffixes=("_human", "_sim"),
    )
    h_scale = joined[[f"{c}_human" for c in cols.scale_columns]].to_numpy(dtype=float)
    s_scale = joined[[f"{c}_sim" for c in cols.scale_columns]].to_numpy(dtype=float)
    h_item = joined[[f"{c}_human" for c in cols.item_columns]].to_numpy(dtype=float)
    s_item = joined[[f"{c}_sim" for c in cols.item_columns]].to_numpy(dtype=float)
    return {
        "ScaleRMSE": np.sqrt(np.mean((h_scale - s_scale) ** 2, axis=1)),
        "ItemMAE": np.mean(np.abs(h_item - s_item), axis=1),
        "ItemEMR": np.mean(h_item == s_item, axis=1),
        "respondent_id": joined["respondent_id"].to_numpy(),
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(values.std(ddof=1)), float(lo), float(hi)


def _paired_effect(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Paired contrast a-b: mean delta, bootstrap CI, and Cohen's d_z."""
    diff = a - b
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else float("nan")
    return float(diff.mean()), float(lo), float(hi), dz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/uncertainty_summary.csv"))
    parser.add_argument("--deltas-out", type=Path, default=Path("results/analysis/uncertainty_deltas.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    metrics = ["ScaleRMSE", "ItemMAE", "ItemEMR"]
    ci_rows: list[dict] = []
    delta_rows: list[dict] = []

    for dataset in ["anket_tr", "bangladesh_sns"]:
        settings = _dataset_settings(dataset)
        cols = _columns(settings)

        # Load both arms from the SADP run for human/train (identical split per dataset).
        sadp_dir = REPO_ROOT / "results" / "runs" / "experiment" / RUNS[(dataset, "SADP")]
        bpp_dir = REPO_ROOT / "results" / "runs" / "experiment" / RUNS[(dataset, "BPP")]

        human, train = _load_run(sadp_dir, cols)
        human = _recompute_scale_means(human, cols)
        train = _impute_train_items(train, cols)
        train = _recompute_scale_means(train, cols)

        sadp_sim = _recompute_scale_means(
            pd.read_csv(sadp_dir / "simulation" / "simulated_responses.csv"), cols
        )
        bpp_sim = _recompute_scale_means(
            pd.read_csv(bpp_dir / "simulation" / "simulated_responses.csv"), cols
        )
        hc_dir = REPO_ROOT / "results" / "runs" / "experiment" / RUNS[(dataset, "HC")]
        hc_sim = _recompute_scale_means(
            pd.read_csv(hc_dir / "simulation" / "simulated_responses.csv"), cols
        )
        pc_sim = _baseline_profile_center(human, train, cols)

        arms = {
            "SADP": _per_respondent_metrics(human, sadp_sim, cols),
            "BPP": _per_respondent_metrics(human, bpp_sim, cols),
            "HC": _per_respondent_metrics(human, hc_sim, cols),
            "Profile-center": _per_respondent_metrics(human, pc_sim, cols),
        }
        n = len(human)

        for arm, mvecs in arms.items():
            for metric in metrics:
                mean, sd, lo, hi = _bootstrap_ci(mvecs[metric], rng)
                ci_rows.append(
                    {
                        "dataset": dataset,
                        "n": n,
                        "arm": arm,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )

        # Effect sizes for key contrasts (aligned per respondent_id).
        def _align(x: dict, y: dict, key: str):
            dfx = pd.DataFrame({"respondent_id": x["respondent_id"], "x": x[key]})
            dfy = pd.DataFrame({"respondent_id": y["respondent_id"], "y": y[key]})
            m = dfx.merge(dfy, on="respondent_id")
            return m["x"].to_numpy(), m["y"].to_numpy()

        for contrast, (a_name, b_name) in {
            "SADP_vs_BPP": ("SADP", "BPP"),
            "SADP_vs_ProfileCenter": ("SADP", "Profile-center"),
            "SADP_vs_HC": ("SADP", "HC"),
            "BPP_vs_HC": ("BPP", "HC"),
        }.items():
            for metric in metrics:
                a, b = _align(arms[a_name], arms[b_name], metric)
                delta, lo, hi, dz = _paired_effect(a, b, rng)
                delta_rows.append(
                    {
                        "dataset": dataset,
                        "n": len(a),
                        "contrast": contrast,
                        "metric": metric,
                        "mean_delta": delta,
                        "delta_ci_lo": lo,
                        "delta_ci_hi": hi,
                        "cohens_dz": dz,
                    }
                )

    ci_df = pd.DataFrame(ci_rows)
    delta_df = pd.DataFrame(delta_rows)

    print(f"\n=== Bootstrap 95pct CIs (per-respondent metric means, B={N_BOOT}) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ci_df.round(4).to_string(index=False))
    print("\n=== Paired contrasts (mean delta, 95% CI, Cohen's d_z) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(delta_df.round(4).to_string(index=False))

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ci_df.round(6).to_csv(out, index=False)
    deltas_out = args.deltas_out if args.deltas_out.is_absolute() else REPO_ROOT / args.deltas_out
    delta_df.round(6).to_csv(deltas_out, index=False)
    print(f"\nWrote {out}\nWrote {deltas_out}")


if __name__ == "__main__":
    main()
