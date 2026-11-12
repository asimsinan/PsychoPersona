"""Canonical run index: one run directory per (dataset, arm, model, split).

Runs accumulate over a revision -- superseded prompts, refitted profile layers, retries
after transient API failures. Every downstream analysis needs to agree on *which* run
represents a cell of the design, so that choice is made once, here, by explicit rules
rather than implicitly by whichever glob happened to match.

Rules, in order:
1. Only runs that completed evaluation with no failed step are eligible.
2. Only runs on the split the paper reports, ``split.random_state == 42``. The
   repeated-split analysis generates runs on seeds 101-110 that are otherwise
   indistinguishable, and because they are more recent than the seed-42 runs they would
   win rule 5 and silently become canonical for the main tables.
3. Profiles must be fitted on scale means alone, with no field that the prompt also
   prints as respondent context. This applies to both datasets. A prompt-visible
   mixture feature makes the profile partly recoverable from the prompt, which turns
   profile recovery into a test of whether the model echoes its own input.
4. Bangladesh SADP must be at or after the prompt alignment that added the two missing
   English instruction lines; earlier runs are excluded. The comparison is on the date
   prefix, so runs from any later day qualify.
5. Only runs matching the requested profile assignment, ``oracle`` by default. The
   cold-start runs regenerate every profile-conditioned cell under predicted
   membership, so they are more recent than their oracle twins and would win rule 6
   and silently replace them in the main tables. The two settings are indexed
   separately and compared side by side, never mixed inside one cell.
6. Among the remaining candidates for a cell, the most recent is canonical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "results" / "runs" / "experiment"
ARMS = {"demographic_only_v1": "DOP", "v3_tr_gpt_b": "BPP", "v3_tr_gpt_d": "FPNP",
        "v3_tr_gpt_e": "BBP", "direct_stats_v1": "SADP",
        "direct_stats_two_pass_v1": "TPSAP", "v4_tr_hybrid_b": "HC"}
MODELS = {"openai/gpt-5.4": "GPT-5.4", "anthropic/claude-opus-4.6": "Claude Opus 4.6",
          "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
          # open-weight families added in the round-2 revision
          "qwen/qwen3.8-27b": "Qwen3.8 27B",
          "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash"}
# Bangladesh SADP prompt gained two instruction lines on this date; earlier runs are stale.
SADP_ALIGNED_FROM = "20260907"
# The split the paper reports. Repeated-split runs use 101-110 and must never be
# eligible here, however recent they are.
PAPER_SPLIT_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--assignment", choices=("oracle", "coldstart"), default="oracle",
                    help="profile assignment to index; the two are never mixed")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path("results/analysis/run_index.csv") if args.assignment == "oracle" \
            else Path("results/analysis/coldstart_run_index.csv")

    rows = []
    for d in sorted(RUNS.iterdir()):
        mf, ev = d / "run_manifest.json", d / "evaluation" / "evaluation_report.json"
        if not (mf.exists() and ev.exists()):
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("failed_step") or "evaluation" not in m.get("completed_steps", []):
            continue
        cfg = m["config"]
        ds = cfg.get("dataset")
        ds = ds.get("dataset_id") if isinstance(ds, dict) else "anket_tr"
        pv = cfg["simulation"]["prompt_version"]
        if pv not in ARMS:
            continue
        model = MODELS.get(m.get("model_id"))
        if model is None:
            continue
        scs = cfg["simulation"].get("self_consistency_samples") or 1
        arm = "SCP" if scs > 1 and pv == "v3_tr_gpt_b" else ARMS[pv]
        feats = cfg["profiles"]["feature_columns"]

        # rule 7: a run that enables reasoning is a diagnostic, not a benchmark run.
        # The benchmark's open-weight runs disable it and the hosted runs do not set
        # it, so without this rule a later diagnostic on the paper's split would win
        # rule 6 and replace the reported cell.
        if ((cfg.get("lmstudio") or {}).get("reasoning") or {}).get("enabled") is True:
            continue
        # rule 8: a run under an alternative profile construction (k-means profiles or
        # soft membership) is a sensitivity run and never a benchmark cell.
        prof = cfg.get("profiles") or {}
        if prof.get("method", "gmm") != "gmm" or prof.get("membership", "hard") != "hard":
            continue
        # rule 2: only the split the paper reports
        if (cfg.get("split") or {}).get("random_state") != PAPER_SPLIT_SEED:
            continue
        # rule 3: no prompt-visible field may be a mixture feature, on either dataset
        if any("usage" in f for f in feats):
            continue
        # rule 5: oracle and cold-start runs are indexed separately. Without this the
        # cold-start runs, being the most recent, would win rule 6 for every
        # profile-conditioned cell and replace the oracle results the tables report.
        if (cfg.get("profiles") or {}).get("assignment", "oracle") != args.assignment:
            continue
        # rule 4: Bangladesh SADP must be at or after the prompt alignment date.
        # Compared on the date prefix rather than with startswith, which matched only
        # the alignment day itself and silently excluded every later run.
        if (ds == "bangladesh_sns" and arm == "SADP"
                and d.name[:len(SADP_ALIGNED_FROM)] < SADP_ALIGNED_FROM):
            continue

        rows.append({"dataset": ds, "arm": arm, "model": model,
                     "split": m["evaluation_split"], "run": d.name,
                     "P": len(pd.read_csv(d / "profiles" / "profile_summary.csv"))})

    frame = pd.DataFrame(rows).sort_values("run")
    canonical = (frame.groupby(["dataset", "arm", "model", "split"], as_index=False).last())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(args.out, index=False)

    order = ["DOP", "BPP", "FPNP", "BBP", "SCP", "SADP", "TPSAP", "HC"]
    print(f"eligible runs {len(frame)} -> canonical cells {len(canonical)}")
    for ds in ["anket_tr", "bangladesh_sns"]:
        sub = canonical[canonical.dataset == ds]
        print(f"\n{ds}  (P={sorted(sub.P.unique())})")
        piv = (sub.assign(cell=1).pivot_table(index="arm", columns="split",
               values="cell", aggfunc="sum", fill_value=0)
               .reindex([a for a in order if a in set(sub.arm)]))
        print(piv.to_string(), "   (6 = 3 models x 1 split each; expect 3 per split)")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
