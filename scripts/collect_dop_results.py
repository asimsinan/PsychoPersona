"""Collect DOP (demographic_only_v1) run results into one summary table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUNS_ROOT = Path("results/runs/experiment")

MODEL_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}


def main() -> None:
    rows = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        if config["simulation"]["prompt_version"] != "demographic_only_v1":
            continue
        run_dir = manifest_path.parent
        report_path = run_dir / "evaluation" / "evaluation_report.json"
        if not report_path.exists():
            print(f"skipping incomplete run {run_dir}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stability_path = run_dir / "stability" / "stability_summary.csv"
        stability = (
            pd.read_csv(stability_path)["correlation_rmse"].mean()
            if stability_path.exists()
            else float("nan")
        )
        indiv = report["individual_level_metrics"]
        rows.append(
            {
                "run_dir": run_dir.name,
                "dataset": config["dataset"]["dataset_id"],
                "split": config["simulation"]["evaluation_split"],
                "model": MODEL_LABELS.get(
                    config["lmstudio"]["model_id"], config["lmstudio"]["model_id"]
                ),
                "corr_rmse": report["correlation_rmse"],
                "within_rmse": report["within_profile_scale_rmse"],
                "recovery": report["profile_recovery_accuracy"],
                "unique": report["response_pattern_diversity"]["simulated_unique_ratio"],
                "stability": stability,
                "scale_rmse": indiv["respondent_scale_rmse_mean"],
                "item_mae": indiv["respondent_item_mae_mean"],
                "item_emr": indiv["respondent_item_exact_match_mean"],
                "full_vector_emr": indiv["respondent_item_all_match_ratio"],
                "n": indiv["aligned_respondent_count"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(["dataset", "split", "model"])
    out_path = Path("results/analysis/dop_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.to_string(index=False))
    print(f"\nwritten to {out_path} ({len(frame)} runs)")


if __name__ == "__main__":
    main()
