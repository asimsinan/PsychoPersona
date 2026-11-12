"""Profile-count (P) reporting and validation-split bootstrap contrasts.

This script adds two analyses requested for the BRM revision that the main
``scripts/compute_uncertainty.py`` does not cover:

1. **Profile-count P and baseline reporting.** For each dataset we report the size
   P of the discovered profile catalog (the number of candidate profiles a
   respondent is matched against). Profile Recovery Accuracy (PRA) is a 1-of-P
   classification metric, so P fixes the uniform chance level (1/P). Because the
   discovered profiles can be strongly imbalanced, 1/P alone understates what a
   trivial predictor achieves, so we also report the per-profile training sizes and
   the majority-profile baseline: the PRA obtained by assigning every held-out
   respondent to the largest training profile.

2. **Validation-split contrast CIs for Two-Pass Statistics-Aware Prompting
   (TPSAP).** TPSAP is a GPT-5.4, Turkish, validation-only ablation, so it is
   not covered by the final-test contrasts in ``compute_uncertainty.py``. Here
   we recompute per-respondent error metrics for TPSAP and the matched
   Statistics-Aware Direct Prompting (SADP) run on the *same* validation
   respondents, then bootstrap the paired contrast (TPSAP - SADP) to obtain
   95% CIs and Cohen's d_z, alongside per-arm bootstrap CIs.

The per-respondent computation mirrors ``saykopersona.evaluation`` and reuses
the helpers in ``scripts/compute_uncertainty.py`` so the bootstrap means
reproduce the point estimates already reported in the manuscript.

Usage
-----
    PYTHONPATH=src python3 scripts/compute_validation_contrasts.py \
        --profiles-out results/analysis/profile_counts.csv \
        --contrasts-out results/analysis/validation_contrasts.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import saykopersona.dataset_catalog as catalog

# Reuse the per-respondent + bootstrap helpers from the main uncertainty script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_uncertainty import (  # noqa: E402
    N_BOOT,
    SEED,
    _bootstrap_ci,
    _columns,
    _paired_effect,
    _per_respondent_metrics,
    _recompute_scale_means,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

def _profile_runs() -> dict[str, str]:
    """One representative canonical run per dataset, read from the run index.

    Pinning run identifiers here is how the Bangladesh profile count and majority
    baseline stayed on a superseded layer through an entire revision: the index moved,
    the constant did not. Reading the index keeps this script on the same runs as every
    other analysis.
    """
    import pandas as _pd
    idx = _pd.read_csv(REPO_ROOT / "results" / "analysis" / "run_index.csv")
    idx = idx[(idx.arm == "SADP") & (idx.model == "GPT-5.4") & (idx.split == "final_test")]
    return {r.dataset: r.run for r in idx.itertuples()}


PROFILE_RUNS: dict[str, str] = _profile_runs()

# Validation-split ablation contrast: TPSAP vs SADP (GPT-5.4, Turkish).
# Both runs share the identical validation respondents (full overlap).
VALIDATION_CONTRAST = {
    "dataset": "anket_tr",
    "model": "GPT-5.4",
    "split": "validation",
    "arms": {
        "TPSAP": "20260413T103218157233Z-837a65a4",
        "SADP": "validation-gpt54-direct-stats-v1-20260412T192427Z",
    },
}


def _dataset_settings(name: str):
    if name == "anket_tr":
        return catalog.default_dataset_settings()
    if name == "bangladesh_sns":
        return catalog.bangladesh_sns_dataset_settings()
    raise SystemExit(f"Unknown dataset '{name}'.")


def _profile_count(run_dir: Path) -> int:
    """Size P of the discovered profile catalog for a run."""
    summary = run_dir / "profiles" / "profile_summary.csv"
    if summary.exists():
        return int(len(pd.read_csv(summary)))
    train_pa = pd.read_csv(run_dir / "profiles" / "train_profile_assignments.csv")
    return int(train_pa["profile_id"].nunique())


def _profile_sizes(run_dir: Path) -> pd.Series:
    """Training-split respondent count per profile id."""
    train_pa = pd.read_csv(run_dir / "profiles" / "train_profile_assignments.csv")
    return train_pa["profile_id"].value_counts().sort_index()


def _pra_labels(run_dir: Path, split: str, settings) -> tuple[pd.Series, pd.Series]:
    """Held-out labels in the label space that Profile Recovery Accuracy actually uses.

    ``evaluation._profile_recovery_accuracy`` does *not* score against the stored
    mixture-model assignments. It rebuilds one centroid per stored profile id from the
    held-out humans' scale means, then relabels every record by nearest standardized
    centroid. That space omits the usage-hours feature the mixture model was fitted on,
    so the two labelings disagree substantially (0.34 agreement on the Turkish
    final-test split). Any baseline meant to be read against PRA must be computed here,
    not on the stored assignments.

    Returns (stored_labels, pra_labels) aligned on the held-out records.
    """
    scale_columns = [scale.mean_column for scale in settings.scales]
    assignments = pd.read_csv(run_dir / "profiles" / f"{split}_profile_assignments.csv")
    human = pd.read_csv(run_dir / "data" / f"{split}_split.csv")
    for scale in settings.scales:
        human[scale.mean_column] = human[list(scale.item_columns)].mean(axis=1)
    merged = assignments[["respondent_id", "profile_id"]].merge(
        human[["respondent_id", *scale_columns]], on="respondent_id"
    )
    centroids = merged.groupby("profile_id")[scale_columns].mean()
    spreads = merged[scale_columns].std(ddof=1).replace(0.0, 1.0)
    centroid_vectors = centroids.div(spreads, axis=1).to_numpy(dtype=float)
    labels = centroids.index.to_numpy()
    points = merged[scale_columns].to_numpy(dtype=float) / spreads.to_numpy(dtype=float)
    nearest = ((points[:, None, :] - centroid_vectors[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return merged["profile_id"], pd.Series(labels[nearest], index=merged.index)


def _majority_profile_baseline(
    run_dir: Path, split: str, settings
) -> tuple[int, float, float, int, float]:
    """Majority-profile baselines for PRA, computed in PRA's own label space.

    Returns (n, train_chosen_pra, oracle_pra, majority_profile_id, label_agreement).

    ``train_chosen_pra`` always predicts the profile that is largest in the *training*
    split, so it uses no held-out information and is the legitimate baseline.
    ``oracle_pra`` is the largest class share among the held-out PRA labels and is an
    upper bound no deployable predictor can be assumed to reach.
    """
    sizes = _profile_sizes(run_dir)
    majority_id = int(sizes.idxmax())
    stored, pra = _pra_labels(run_dir, split, settings)
    n = len(pra)
    counts = pra.value_counts()
    return (
        n,
        float((pra == majority_id).mean()),
        float(counts.max() / n),
        majority_id,
        float((stored.to_numpy() == pra.to_numpy()).mean()),
    )


def _load_validation(run_dir: Path):
    """Validation human responses joined with their profile assignments."""
    human = pd.read_csv(run_dir / "data" / "validation_split.csv")
    pa = pd.read_csv(run_dir / "profiles" / "validation_profile_assignments.csv")
    human = human.merge(pa[["respondent_id", "profile_id"]], on="respondent_id", how="left")
    return human


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-out", type=Path, default=Path("results/analysis/profile_counts.csv")
    )
    parser.add_argument(
        "--contrasts-out", type=Path, default=Path("results/analysis/validation_contrasts.csv")
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    metrics = ["ScaleRMSE", "ItemMAE", "ItemEMR"]

    # ---- 1. Profile-count P reporting --------------------------------------
    profile_rows: list[dict] = []
    for dataset, run in PROFILE_RUNS.items():
        run_dir = REPO_ROOT / "results" / "runs" / "experiment" / run
        p = _profile_count(run_dir)
        sizes = _profile_sizes(run_dir)
        settings = _dataset_settings(dataset)
        for split in ("validation", "final_test"):
            n_split, majority_pra, oracle_pra, majority_id, agreement = (
                _majority_profile_baseline(run_dir, split, settings)
            )
            profile_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "profile_count_P": p,
                    "pra_chance_level": round(1.0 / p, 6) if p else float("nan"),
                    "train_profile_sizes": "/".join(str(int(v)) for v in sizes.to_list()),
                    "train_n": int(sizes.sum()),
                    "majority_profile_id": majority_id,
                    "majority_profile_share_train": round(float(sizes.max() / sizes.sum()), 6),
                    "n": n_split,
                    "majority_profile_pra": round(majority_pra, 6),
                    "largest_class_pra_oracle": round(oracle_pra, 6),
                    "stored_vs_pra_label_agreement": round(agreement, 6),
                    "run": run,
                }
            )
    profiles_df = pd.DataFrame(profile_rows)

    # ---- 2. Validation-split TPSAP vs SADP contrast ------------------------
    dataset = VALIDATION_CONTRAST["dataset"]
    settings = _dataset_settings(dataset)
    cols = _columns(settings)

    arm_dirs = {
        arm: REPO_ROOT / "results" / "runs" / "experiment" / run
        for arm, run in VALIDATION_CONTRAST["arms"].items()
    }
    # Both arms share the same validation respondents; load human/profile once.
    human = _recompute_scale_means(_load_validation(arm_dirs["SADP"]), cols)

    arm_metrics: dict[str, dict[str, np.ndarray]] = {}
    for arm, d in arm_dirs.items():
        sim = _recompute_scale_means(
            pd.read_csv(d / "simulation" / "simulated_responses.csv"), cols
        )
        arm_metrics[arm] = _per_respondent_metrics(human, sim, cols)

    n = len(human)
    ci_rows: list[dict] = []
    for arm, mvecs in arm_metrics.items():
        for metric in metrics:
            mean, sd, lo, hi = _bootstrap_ci(mvecs[metric], rng)
            ci_rows.append(
                {
                    "dataset": dataset,
                    "model": VALIDATION_CONTRAST["model"],
                    "split": VALIDATION_CONTRAST["split"],
                    "n": n,
                    "arm": arm,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    def _align(x: dict, y: dict, key: str):
        dfx = pd.DataFrame({"respondent_id": x["respondent_id"], "x": x[key]})
        dfy = pd.DataFrame({"respondent_id": y["respondent_id"], "y": y[key]})
        m = dfx.merge(dfy, on="respondent_id")
        return m["x"].to_numpy(), m["y"].to_numpy()

    delta_rows: list[dict] = []
    for metric in metrics:
        a, b = _align(arm_metrics["TPSAP"], arm_metrics["SADP"], metric)
        delta, lo, hi, dz = _paired_effect(a, b, rng)
        delta_rows.append(
            {
                "dataset": dataset,
                "model": VALIDATION_CONTRAST["model"],
                "split": VALIDATION_CONTRAST["split"],
                "n": len(a),
                "contrast": "TPSAP_vs_SADP",
                "metric": metric,
                "mean_delta": delta,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "cohens_dz": dz,
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    delta_df = pd.DataFrame(delta_rows)
    contrasts_df = pd.concat(
        [ci_df.assign(kind="per_arm_ci"), delta_df.assign(kind="paired_contrast")],
        ignore_index=True,
    )

    # ---- Report ------------------------------------------------------------
    print("=== Profile-count P (PRA is 1-of-P; chance level = 1/P) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(profiles_df.to_string(index=False))

    print(
        f"\n=== Validation per-arm bootstrap 95pct CIs "
        f"({VALIDATION_CONTRAST['model']} {dataset}, split={VALIDATION_CONTRAST['split']}, "
        f"n={n}, B={N_BOOT}) ==="
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ci_df.round(4).to_string(index=False))

    print("\n=== Validation paired contrast TPSAP - SADP (mean delta, 95% CI, Cohen's d_z) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(delta_df.round(4).to_string(index=False))

    profiles_out = (
        args.profiles_out if args.profiles_out.is_absolute() else REPO_ROOT / args.profiles_out
    )
    contrasts_out = (
        args.contrasts_out if args.contrasts_out.is_absolute() else REPO_ROOT / args.contrasts_out
    )
    profiles_out.parent.mkdir(parents=True, exist_ok=True)
    contrasts_out.parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(profiles_out, index=False)
    contrasts_df.round(6).to_csv(contrasts_out, index=False)
    print(f"\nWrote {profiles_out}\nWrote {contrasts_out}")


if __name__ == "__main__":
    main()
