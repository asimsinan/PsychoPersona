"""Generation under alternative profile constructions (Reviewer 1, comment 3).

Sixteen GPT-5.4 runs, BPP and SADP on both datasets and both evaluation splits, repeat
the benchmark with one element of the profile layer changed.

kmeans  profiles from k-means in place of the Gaussian mixture, three profiles, oracle
        assignment. Compared with the oracle run of the same arm and split.
soft    cold-start assignment with the classifier's membership probabilities passed to
        the prompt in place of one predicted profile. Compared with the hard cold-start
        run of the same arm and split.

Each run is scored by the benchmark's evaluation code against the same held-out
respondents as its reference run, so respondent-level error is compared respondent by
respondent, with a bootstrap interval over respondents. Profile recovery and
within-profile error are scored in the label space of the run's own partition, next to
the majority rate and the profile-center baseline in that label space, because their
level depends on the partition (see ``r2_profile_sensitivity.py``).

The expectation for the soft runs was stated before they were launched: no difference
from hard cold-start, because the classifier's probabilities sit close to the base rates.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_alt_profile_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_nonllm_baselines import (
    _columns, _dataset_settings, _evaluate, _impute_train_items, _recompute_scale_means,
    baseline_profile_center,
)
from r2_profile_sensitivity import recovery_labels

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
RUNS = REPO / "results" / "runs" / "experiment"
FIRST = "20261112T08"          # nothing earlier is an alternative-profile run
ARMS = {"v3_tr_gpt_b": "BPP", "direct_stats_v1": "SADP"}
METRICS = ["RespItemMAE", "RespScaleRMSE", "ItemEMR", "CorrRMSE", "WithinRMSE", "PRA", "SimUniqueRatio"]
REPS = 5000
SEED = 20261112


def load(run: str, cols):
    d = RUNS / run
    split = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))["evaluation_split"]
    human = pd.read_csv(d / "data" / f"{split}_split.csv").merge(
        pd.read_csv(d / "profiles" / f"{split}_profile_assignments.csv")[["respondent_id", "profile_id"]],
        on="respondent_id")
    human = _recompute_scale_means(human, cols)
    train = pd.read_csv(d / "data" / "train_split.csv").merge(
        pd.read_csv(d / "profiles" / "train_profile_assignments.csv")[["respondent_id", "profile_id"]],
        on="respondent_id")
    train = _recompute_scale_means(_impute_train_items(train, cols), cols)
    sim = pd.read_csv(d / "simulation" / "simulated_responses.csv").set_index("respondent_id")
    sim = _recompute_scale_means(sim.loc[human.respondent_id].reset_index(), cols)
    sim["profile_id"] = human.profile_id.values
    return human, sim, train


def describe(run: str, cols) -> tuple[dict, np.ndarray, pd.Series]:
    human, sim, train = load(run, cols)
    row = _evaluate(human, sim, cols)
    mae = np.abs(sim[cols.item_columns].to_numpy(dtype=float)
                 - human[cols.item_columns].to_numpy(dtype=float)).mean(axis=1)
    if human.profile_id.nunique() > 1:
        row["majority"] = float(pd.Series(recovery_labels(human, human, cols.scale_columns))
                                .value_counts(normalize=True).max())
    else:
        row["majority"] = float("nan")
    center = _evaluate(human, baseline_profile_center(human, train, cols), cols)
    row["center_item_mae"] = center["RespItemMAE"]
    row["center_recovery"] = center["PRA"]
    row["profiles_assigned"] = int(human.profile_id.nunique())
    return row, mae, human.respondent_id


def main() -> None:
    rng = np.random.default_rng(SEED)
    oracle = pd.read_csv(ANALYSIS / "run_index.csv")
    cold = pd.read_csv(ANALYSIS / "coldstart_run_index.csv")
    rows = []
    for d in sorted(RUNS.iterdir()):
        manifest = d / "run_manifest.json"
        if d.name < FIRST or not manifest.exists():
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        cfg = m["config"]
        prof = cfg.get("profiles") or {}
        variant = "kmeans" if prof.get("method", "gmm") == "kmeans" else \
            "soft" if prof.get("membership", "hard") == "soft" else None
        if variant is None or m.get("failed_step") or "evaluation" not in m.get("completed_steps", []):
            continue
        ds = cfg["dataset"]["dataset_id"] if isinstance(cfg.get("dataset"), dict) else "anket_tr"
        arm = ARMS[cfg["simulation"]["prompt_version"]]
        split = cfg["simulation"]["evaluation_split"]
        index = oracle if variant == "kmeans" else cold
        ref = index[(index.dataset == ds) & (index.arm == arm) & (index.model == "GPT-5.4") & (index.split == split)]
        if len(ref) != 1:
            raise SystemExit(f"no unique reference run for {ds} {arm} {split} {variant}")
        cols = _columns(_dataset_settings(ds))
        new, new_mae, new_ids = describe(d.name, cols)
        old, old_mae, old_ids = describe(ref.run.iloc[0], cols)
        if list(new_ids) != list(old_ids):
            raise SystemExit(f"{d.name}: held-out respondents differ from the reference run")
        diff = new_mae - old_mae
        boot = diff[rng.integers(0, len(diff), size=(REPS, len(diff)))].mean(axis=1)
        row = {"dataset": ds, "variant": variant, "arm": arm, "split": split, "n": len(diff),
               "run": d.name, "reference_run": ref.run.iloc[0]}
        for k in [*METRICS, "majority", "center_item_mae", "center_recovery", "profiles_assigned"]:
            row[f"{k}_reference"] = old[k]
            row[f"{k}_variant"] = new[k]
        row["item_mae_difference"] = float(diff.mean())
        row["item_mae_difference_lo"] = float(np.percentile(boot, 2.5))
        row["item_mae_difference_hi"] = float(np.percentile(boot, 97.5))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["variant", "dataset", "arm", "split"]).reset_index(drop=True)
    out.round(6).to_csv(ANALYSIS / "r2_alt_profile_runs.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"alternative-profile runs found: {len(out)} of 16")
    key = ["dataset", "arm", "split"]
    for variant, title in (("kmeans", "k-means profiles against the mixture, oracle assignment"),
                           ("soft", "soft against hard membership, cold-start assignment")):
        g = out[out.variant == variant].set_index(key)
        print(f"\n=== {title} ===")
        print("item MAE (difference is variant minus reference, paired by respondent, 95% bootstrap interval)")
        print(g[["n", "RespItemMAE_reference", "RespItemMAE_variant", "item_mae_difference",
                 "item_mae_difference_lo", "item_mae_difference_hi", "center_item_mae_reference",
                 "center_item_mae_variant"]].round(3).to_string())
        print("other metrics, reference then variant")
        show = ["RespScaleRMSE", "CorrRMSE", "WithinRMSE", "PRA", "majority", "center_recovery", "SimUniqueRatio"]
        print(g[[f"{k}_{s}" for k in show for s in ("reference", "variant")]]
              .rename(columns=lambda c: c.replace("_reference", " ref").replace("_variant", " var")
                      .replace("RespScaleRMSE", "scaleRMSE").replace("SimUniqueRatio", "unique")
                      .replace("WithinRMSE", "within").replace("center_recovery", "centerPRA")).round(3).to_string())
    # does SADP still sit below BPP, and both above the baseline, under each construction?
    print("\n=== SADP minus BPP on item MAE, and each arm minus profile-center, by construction ===")
    w = out.pivot_table(index=["variant", "dataset", "split"], columns="arm",
                        values=["RespItemMAE_reference", "RespItemMAE_variant",
                                "center_item_mae_reference", "center_item_mae_variant"])
    t = pd.DataFrame({
        "SADP-BPP reference": w[("RespItemMAE_reference", "SADP")] - w[("RespItemMAE_reference", "BPP")],
        "SADP-BPP variant": w[("RespItemMAE_variant", "SADP")] - w[("RespItemMAE_variant", "BPP")],
        "BPP-center reference": w[("RespItemMAE_reference", "BPP")] - w[("center_item_mae_reference", "BPP")],
        "BPP-center variant": w[("RespItemMAE_variant", "BPP")] - w[("center_item_mae_variant", "BPP")],
        "SADP-center reference": w[("RespItemMAE_reference", "SADP")] - w[("center_item_mae_reference", "SADP")],
        "SADP-center variant": w[("RespItemMAE_variant", "SADP")] - w[("center_item_mae_variant", "SADP")]})
    print(t.round(3).to_string())
    print(f"\nwrote r2_alt_profile_runs.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
