"""Every metric of every main-grid run in one table.

One row per run of the paper's split (seed 42): eight arms on the three hosted families
and four arms on the two open-weight families, validation and final test, under oracle
and cold-start assignment. Values are read from each run's own evaluation report, which
is what the manuscript's tables were built from, so nothing is recomputed here except the
majority-profile rate, which the report does not store.

Output: results/analysis/r2_main_grid.csv. The appendix tables and the results text take
their main-grid numbers from this file.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_main_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import r2_common as C
from compute_nonllm_baselines import _columns, _dataset_settings, _recompute_scale_means
from r2_profile_sensitivity import recovery_labels

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
RUNS = REPO / "results" / "runs" / "experiment"


def describe(run: str, ds: str) -> dict:
    d = RUNS / run
    rep = json.loads((d / "evaluation" / "evaluation_report.json").read_text(encoding="utf-8"))
    ind = rep["individual_level_metrics"]
    div = rep["response_pattern_diversity"]
    row = {"corr_rmse": rep["correlation_rmse"], "within_rmse": rep["within_profile_scale_rmse"],
           "recovery": rep["profile_recovery_accuracy"], "scale_rmse": ind["respondent_scale_rmse_mean"],
           "item_mae": ind["respondent_item_mae_mean"], "item_emr": ind["respondent_item_exact_match_mean"],
           "full_vector_emr": ind["respondent_item_all_match_ratio"], "unique": div["simulated_unique_ratio"],
           "human_unique": div["human_unique_ratio"], "n": ind["aligned_respondent_count"]}
    stab = d / "stability" / "stability_summary.csv"
    row["stability"] = float(pd.read_csv(stab).correlation_rmse.mean()) if stab.exists() else float("nan")
    cols = _columns(_dataset_settings(ds))
    human, _, _ = C.load_run(run, ds)
    human = _recompute_scale_means(human, cols)
    row["profiles_assigned"] = int(human.profile_id.nunique())
    row["majority"] = (float(pd.Series(recovery_labels(human, human, cols.scale_columns)).value_counts(normalize=True).max())
                       if row["profiles_assigned"] > 1 else float("nan"))
    calls = list((d / "llm_archive").glob("*chat_completions.json"))
    row["api_calls"] = len(calls)
    return row


def main() -> None:
    rows = []
    for assignment, name in (("oracle", "run_index.csv"), ("coldstart", "coldstart_run_index.csv")):
        idx = pd.read_csv(ANALYSIS / name)
        for r in idx.itertuples():
            rows.append({"dataset": r.dataset, "split": r.split, "assignment": assignment, "arm": r.arm,
                         "model": r.model, **describe(r.run, r.dataset), "run": r.run})
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "r2_main_grid.csv", index=False)
    print(f"wrote r2_main_grid.csv: {len(out)} runs")
    print(out.groupby(["assignment", "dataset", "split"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
