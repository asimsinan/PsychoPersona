"""Cost, dates and serving path of the round-2 generation runs.

One row per run generated for the round-2 revision: which block it belongs to, how many
calls it made, the tokens and the amount billed by the provider, when it ran, and which
host and service tier served it. Section 3.10 of the manuscript reports these totals,
and the serving path is what the paper discloses in place of a drift test.

Usage
-----
    PYTHONPATH=src python scripts/r2_generation_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs" / "experiment"
FIRST_RUN = "20261111T2029"      # the pilot; nothing earlier belongs to round 2
MODELS = {"openai/gpt-5.4": "GPT-5.4", "anthropic/claude-opus-4.6": "Claude Opus 4.6",
          "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
          "qwen/qwen3.8-27b": "Qwen3.8 27B",
          "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash"}
ARMS = {"demographic_only_v1": "DOP", "v3_tr_gpt_b": "BPP", "v3_tr_gpt_d": "FPNP",
        "v3_tr_gpt_e": "BBP", "direct_stats_v1": "SADP", "direct_stats_two_pass_v1": "TPSAP"}
OPEN = ("Qwen3.8 27B", "DeepSeek V4.1 Flash")


def describe(run: Path) -> dict | None:
    manifest = run / "run_manifest.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if m.get("failed_step") or "evaluation" not in m.get("completed_steps", []):
        return None
    cfg = m["config"]
    sim = cfg["simulation"]
    arm = "HC" if (cfg.get("calibration") or {}).get("enabled") else ARMS.get(sim["prompt_version"])
    if arm == "BPP" and (sim.get("self_consistency_samples") or 1) > 1:
        arm = "SCP"
    model = MODELS.get(cfg["lmstudio"]["model_id"])
    ds = cfg.get("dataset")
    ds = ds.get("dataset_id") if isinstance(ds, dict) else "anket_tr"
    seed = (cfg.get("split") or {}).get("random_state", 42)
    assignment = (cfg.get("profiles") or {}).get("assignment") or "oracle"
    if ((cfg["lmstudio"].get("reasoning") or {}).get("enabled")) is True:
        block = "diagnostic, reasoning enabled"
    elif (cfg.get("profiles") or {}).get("method", "gmm") != "gmm":
        block = "alternative profiles, k-means"
    elif (cfg.get("profiles") or {}).get("membership", "hard") != "hard":
        block = "alternative profiles, soft membership"
    elif seed != 42:
        block = "repeated splits"
    elif assignment == "coldstart":
        block = "main grid, cold-start"
    else:
        block = "main grid, oracle"

    calls = prompt = completion = retried = 0
    cost = 0.0
    hosts, tiers, stamps = set(), set(), []
    for f in (run / "llm_archive").glob("*chat_completions.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        stamps.append(j["archived_at"])
        r = j.get("response") or {}
        if j.get("error") or j.get("response_status") not in (200, None) or "choices" not in r:
            retried += 1
            continue
        u = r.get("usage") or {}
        calls += 1
        prompt += u.get("prompt_tokens") or 0
        completion += u.get("completion_tokens") or 0
        cost += u.get("cost") or 0.0
        hosts.add(str(r.get("provider")))
        tiers.add(str(r.get("service_tier")))
    return {"run": run.name, "block": block, "dataset": ds, "model": model, "arm": arm,
            "split": sim["evaluation_split"], "seed": seed, "assignment": assignment,
            "calls": calls, "retried_calls": retried, "prompt_tokens": prompt,
            "completion_tokens": completion, "cost_usd": round(cost, 6),
            "first_call_utc": min(stamps)[:19], "last_call_utc": max(stamps)[:19],
            "minutes": round((pd.Timestamp(max(stamps)) - pd.Timestamp(min(stamps))).total_seconds() / 60, 2),
            "hosts": "; ".join(sorted(hosts)), "service_tiers": "; ".join(sorted(tiers))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/analysis/r2_generation_summary.csv"))
    args = ap.parse_args()
    rows = [d for run in sorted(RUNS.iterdir()) if run.name >= FIRST_RUN and run.name[:2] == "20"
            for d in [describe(run)] if d]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(f"round-2 runs: {len(df)} | calls {df.calls.sum():,} | tokens "
          f"{(df.prompt_tokens.sum() + df.completion_tokens.sum()) / 1e6:.1f} million | "
          f"billed ${df.cost_usd.sum():.2f} | generation time {df.minutes.sum() / 60:.1f} hours")
    print(f"generated between {df.first_call_utc.min()} and {df.last_call_utc.max()} UTC")
    print("\nby model family:")
    g = df.groupby("model").agg(runs=("run", "size"), calls=("calls", "sum"), billed=("cost_usd", "sum"),
                                per_run=("cost_usd", "mean"), hosts=("hosts", lambda s: "; ".join(sorted(set(s)))),
                                tiers=("service_tiers", lambda s: "; ".join(sorted(set(s)))))
    print(g.round(2).to_string())
    print("\nby block:")
    df["family_group"] = df.model.where(~df.model.isin(OPEN), "open-weight")
    print(df.groupby(["block", "family_group"]).agg(runs=("run", "size"), billed=("cost_usd", "sum")).round(2).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
