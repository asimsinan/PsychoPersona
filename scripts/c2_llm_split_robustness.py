"""Repeated-split robustness for the LLM arms (C2, Reviewer 1 comment 2).

Reviewer 1 objects that results resting on a single 80/10/10 split confound a method
with the split it was measured on. ``scripts/split_robustness.py`` answers the non-LLM
half of that question by repeating the split, profile fit, assignment and evaluation
across 50 draws. This script answers the LLM half, which requires regenerating
responses under each split and therefore cannot be done offline.

Design: ten splits (``split.random_state`` 101--110) on both datasets. Round 1 covered
{DOP, BPP, SADP, HC} on GPT-5.4, 80 runs. Round 2 (Reviewer 1, comment 4) adds the other
four arms on GPT-5.4 and the same four arms on Claude Opus 4.6, Gemini 3.1 Pro and two
open-weight models. Every model family is reported separately. Seeds 101--110 keep these
runs disjoint from the seed-42 runs that back the manuscript's main tables, so
``build_run_index.py`` never mixes the two.

Two quantities are reported. First, the between-split dispersion of each arm on each
headline metric, which says how much of a reported number belongs to the split rather
than the method. Second, for each contrast that carries a claim in the paper, the
dispersion of the contrast itself and the proportion of splits in which its sign is
preserved. The second matters more: a contrast can be stable even when both of its
terms move a great deal, because the split perturbs them together.

Usage
-----
    PYTHONPATH=src python3 scripts/c2_llm_split_robustness.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs" / "experiment"

SEEDS = range(101, 111)
MODELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "qwen/qwen3.8-27b": "Qwen3.8 27B",
    "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
}
ARM_ORDER = ["DOP", "BPP", "FPNP", "BBP", "SCP", "SADP", "TPSAP", "HC"]

# prompt_version -> arm label. Hybrid Calibration is identified by calibration.enabled
# rather than by prompt_version, because it shares a prompt with the hybrid arm, and
# Self-Consistency Prompting by its sample count, because it shares BPP's prompt.
ARM_BY_PROMPT = {
    "demographic_only_v1": "DOP",
    "v3_tr_gpt_b": "BPP",
    "v3_tr_gpt_d": "FPNP",
    "v3_tr_gpt_e": "BBP",
    "direct_stats_v1": "SADP",
    "direct_stats_two_pass_v1": "TPSAP",
}

# metric key -> (label, lower_is_better)
METRICS = {
    "correlation_rmse": ("Corr RMSE", True),
    "profile_recovery_accuracy": ("Profile recovery", False),
    "respondent_item_mae_mean": ("Item MAE", True),
    "simulated_unique_ratio": ("Unique ratio", False),
}

# Contrasts written as (A, B); the reported delta is A - B. Each is a comparison the
# manuscript draws a conclusion from.
CONTRASTS = [
    ("SADP", "BPP", "profile statistics beyond the persona card (RQ2)"),
    ("BPP", "DOP", "persona conditioning beyond demographics (RQ5)"),
    ("SADP", "DOP", "statistics-aware prompting beyond demographics (RQ5)"),
    ("HC", "SADP", "calibrated decoding against prompt-only (RQ3)"),
]


def _read_metrics(run_dir: Path) -> dict[str, float] | None:
    report = run_dir / "evaluation" / "evaluation_report.json"
    if not report.exists():
        return None
    r = json.loads(report.read_text(encoding="utf-8"))
    ind = r.get("individual_level_metrics", {})
    div = r.get("response_pattern_diversity", {})
    return {
        "correlation_rmse": r.get("correlation_rmse"),
        "profile_recovery_accuracy": r.get("profile_recovery_accuracy"),
        "respondent_item_mae_mean": ind.get("respondent_item_mae_mean"),
        "simulated_unique_ratio": div.get("simulated_unique_ratio"),
    }


def collect() -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = manifest.get("config", {})
        seed = (cfg.get("split") or {}).get("random_state")
        if seed is None or int(seed) not in SEEDS:
            continue
        model = MODELS.get(manifest.get("model_id"))
        if model is None:
            continue
        # Oracle assignment only; a cold-start run on these seeds would otherwise be
        # pooled with the oracle runs of the same cell.
        if (cfg.get("profiles") or {}).get("assignment", "oracle") != "oracle":
            continue
        # A run that enables reasoning is a diagnostic and is never pooled here.
        if ((cfg.get("lmstudio") or {}).get("reasoning") or {}).get("enabled") is True:
            continue
        # Neither is a run under an alternative profile construction.
        prof = cfg.get("profiles") or {}
        if prof.get("method", "gmm") != "gmm" or prof.get("membership", "hard") != "hard":
            continue
        sim = cfg.get("simulation", {})
        arm = "HC" if (cfg.get("calibration") or {}).get("enabled") else \
            ARM_BY_PROMPT.get(sim.get("prompt_version"))
        if arm == "BPP" and (sim.get("self_consistency_samples") or 1) > 1:
            arm = "SCP"
        if arm is None:
            continue
        metrics = _read_metrics(manifest_path.parent)
        if metrics is None or metrics["correlation_rmse"] is None:
            continue
        ds = cfg.get("dataset")
        ds = ds.get("dataset_id") if isinstance(ds, dict) else "anket_tr"
        rows.append({"dataset": ds, "model": model, "arm": arm, "seed": int(seed),
                     "run": manifest_path.parent.name, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-out", type=Path,
                        default=Path("results/analysis/c2_llm_runs.csv"))
    parser.add_argument("--arms-out", type=Path,
                        default=Path("results/analysis/c2_llm_arm_dispersion.csv"))
    parser.add_argument("--contrasts-out", type=Path,
                        default=Path("results/analysis/c2_llm_contrasts.csv"))
    args = parser.parse_args()

    df = collect()
    if df.empty:
        raise SystemExit("No C2 runs found. Generate seeds 101-110 first.")

    dup = df.duplicated(subset=["dataset", "model", "arm", "seed"]).sum()
    if dup:
        # Keep the most recent run per cell; a repeated cell would otherwise weight
        # one split twice in every dispersion figure below.
        df = df.sort_values("run").drop_duplicates(
            subset=["dataset", "model", "arm", "seed"], keep="last")

    args.runs_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.runs_out, index=False)

    print("=== C2 coverage ===")
    cover = (df.groupby(["dataset", "model", "arm"]).seed.nunique().unstack(fill_value=0)
             .reindex(columns=[a for a in ARM_ORDER if a in df.arm.unique()]))
    print(cover.to_string())
    if dup:
        print(f"  ({dup} duplicate cell(s) collapsed to the most recent run)")

    arm_rows = []
    cells = [(ds, m) for ds in sorted(df.dataset.unique())
             for m in MODELS.values() if ((df.dataset == ds) & (df.model == m)).any()]
    print("\n=== Between-split dispersion of each arm ===")
    for ds, model in cells:
        print(f"\n-- {ds} | {model} --")
        print(f"{'arm':6s} {'metric':17s} {'mean':>7s} {'sd':>7s} {'min':>7s} {'max':>7s} {'range':>7s}")
        for arm in ARM_ORDER:
            sub = df[(df.dataset == ds) & (df.model == model) & (df.arm == arm)]
            if sub.empty:
                continue
            for key, (label, _) in METRICS.items():
                v = sub[key].dropna()
                if v.empty:
                    continue
                arm_rows.append({"dataset": ds, "model": model, "arm": arm, "metric": label,
                                 "n_splits": len(v), "mean": round(v.mean(), 6),
                                 "sd": round(v.std(ddof=1), 6),
                                 "min": round(v.min(), 6), "max": round(v.max(), 6),
                                 "range": round(v.max() - v.min(), 6)})
                print(f"{arm:6s} {label:17s} {v.mean():7.3f} {v.std(ddof=1):7.3f} "
                      f"{v.min():7.3f} {v.max():7.3f} {v.max() - v.min():7.3f}")
    pd.DataFrame(arm_rows).to_csv(args.arms_out, index=False)

    con_rows = []
    print("\n=== Paired contrasts across splits (delta = A - B, paired within split) ===")
    for ds, model in cells:
        print(f"\n-- {ds} | {model} --")
        print(f"{'contrast':14s} {'metric':17s} {'mean':>8s} {'sd':>7s} "
              f"{'min':>7s} {'max':>7s} {'sign kept':>10s}")
        here = df[(df.dataset == ds) & (df.model == model)]
        for a, b, _note in CONTRASTS:
            pa = here[here.arm == a].set_index("seed")
            pb = here[here.arm == b].set_index("seed")
            shared = sorted(set(pa.index) & set(pb.index))
            if not shared:
                continue
            for key, (label, lower_better) in METRICS.items():
                d = (pa.loc[shared, key] - pb.loc[shared, key]).dropna()
                if d.empty:
                    continue
                # "Favours A" means A is better, which flips with the metric's polarity.
                favours_a = (d < 0) if lower_better else (d > 0)
                # Sign preservation is agreement with the direction of the mean
                # contrast, so it reads as "how often a single split reproduces the
                # conclusion the pooled estimate draws". Taking the larger of the two
                # proportions instead would report a contrast as stable while the
                # majority of splits pointed the opposite way to its mean.
                mean_favours_a = (d.mean() < 0) if lower_better else (d.mean() > 0)
                majority = favours_a.mean() if mean_favours_a else (~favours_a).mean()
                con_rows.append({
                    "dataset": ds, "model": model, "contrast": f"{a}-{b}", "metric": label,
                    "n_splits": len(d), "mean_delta": round(d.mean(), 6),
                    "sd": round(d.std(ddof=1), 6), "min": round(d.min(), 6),
                    "max": round(d.max(), 6),
                    "splits_favouring_A": int(favours_a.sum()),
                    "sign_preserved": round(majority, 3)})
                print(f"{a + '-' + b:14s} {label:17s} {d.mean():8.3f} {d.std(ddof=1):7.3f} "
                      f"{d.min():7.3f} {d.max():7.3f} {majority * 100:9.0f}%")
    pd.DataFrame(con_rows).to_csv(args.contrasts_out, index=False)

    print(f"\nwrote {args.runs_out}, {args.arms_out}, {args.contrasts_out}")


if __name__ == "__main__":
    main()
