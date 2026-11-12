"""Generation-run variance (A4) and computational cost (A5) from the run archive.

A4 -- Reviewer 1 asks whether results are sensitive to the particular generation run.
Every experiment already repeats generation three times for the stability sweep, so
between-run dispersion can be reported without new runs. Two accidental replicate pairs
add a second estimate at the level of a whole run rather than a stability sweep.

A5 -- Reviewer 4 asks for computational cost. Token usage is recorded per archived call;
wall-clock is derived from the first and last archived call in each run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs" / "experiment"

REPLICATE_PAIRS = [
    ("bangladesh_sns HC final-test", "20260413T093157879381Z-3440b06e",
     "20260413T093157886393Z-5de10863", "same config, independent runs"),
    ("anket_tr BPP final-test", "finaltest-gpt54-tuned-b-20260412T165727Z",
     "finaltest-claude-opus46-tuned-e-claude-a-20260412T170452Z",
     "identical instruction text, different persona-card serialization (Claude)"),
]


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variance-out", type=Path,
                        default=Path("results/analysis/generation_variance.csv"))
    parser.add_argument("--cost-out", type=Path,
                        default=Path("results/analysis/computational_cost.csv"))
    args = parser.parse_args()

    var_rows, cost_rows = [], []
    for manifest_path in sorted(RUNS_ROOT.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = manifest_path.parent
        cfg = manifest.get("config", {})
        ds = cfg.get("dataset")
        ds = ds.get("dataset_id") if isinstance(ds, dict) else "anket_tr"
        label = {
            "dataset": ds,
            "model": (manifest.get("model_id") or "").split("/")[-1],
            "arm": cfg.get("simulation", {}).get("prompt_version"),
            "calibrated": cfg.get("calibration", {}).get("enabled"),
            "split": manifest.get("evaluation_split"),
            "run": run_dir.name,
        }

        stability = manifest.get("stability_runs") or []
        if len(stability) >= 2:
            corr = np.array([s["correlation_rmse"] for s in stability], dtype=float)
            pra = np.array([s.get("profile_recovery_accuracy", np.nan) for s in stability],
                           dtype=float)
            var_rows.append({
                **label, "source": f"stability sweep (n={len(corr)})",
                "corr_rmse_mean": round(corr.mean(), 6), "corr_rmse_sd": round(corr.std(ddof=1), 6),
                "corr_rmse_range": round(corr.max() - corr.min(), 6),
                "pra_mean": round(np.nanmean(pra), 6), "pra_sd": round(np.nanstd(pra, ddof=1), 6),
                "pra_range": round(np.nanmax(pra) - np.nanmin(pra), 6),
            })

        prompt_tok = completion_tok = calls = 0
        cost_usd = 0.0
        stamps = []
        for f in (run_dir / "llm_archive").glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            usage = (d.get("response") or {}).get("usage") or {}
            if usage:
                prompt_tok += usage.get("prompt_tokens", 0)
                completion_tok += usage.get("completion_tokens", 0)
                # OpenRouter returns the billed amount per call, so monetary cost is
                # recorded rather than estimated from list prices, which drift.
                cost_usd += float(usage.get("cost", 0.0) or 0.0)
                calls += 1
            ts = _parse(str(d.get("archived_at", "")))
            if ts:
                stamps.append(ts)
        if calls:
            minutes = ((max(stamps) - min(stamps)).total_seconds() / 60) if len(stamps) > 1 else 0.0
            cost_rows.append({
                **label, "calls": calls, "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "cost_usd": round(cost_usd, 4),
                "wallclock_minutes": round(minutes, 1),
            })

    var = pd.DataFrame(var_rows)
    cost = pd.DataFrame(cost_rows)
    args.variance_out.parent.mkdir(parents=True, exist_ok=True)
    var.to_csv(args.variance_out, index=False)
    cost.to_csv(args.cost_out, index=False)

    print("=== A4: generation-run variance, dispersion across stability sweeps ===")
    print(f"  runs with a sweep: {len(var)}")
    for metric in ("corr_rmse", "pra"):
        sd, rng = var[f"{metric}_sd"].dropna(), var[f"{metric}_range"].dropna()
        print(f"  {metric:10s} within-run SD: median {sd.median():.4f}  "
              f"p90 {sd.quantile(0.9):.4f}  max {sd.max():.4f}   |   "
              f"range: median {rng.median():.4f}  max {rng.max():.4f}")
    print("\n  accidental whole-run replicate pairs:")
    for name, a, b, note in REPLICATE_PAIRS:
        try:
            ra, rb = (json.loads((RUNS_ROOT / r / "evaluation" / "evaluation_report.json")
                                 .read_text()) for r in (a, b))
        except Exception:
            continue
        print(f"    {name} ({note})")
        print(f"      corr RMSE {ra['correlation_rmse']:.4f} vs {rb['correlation_rmse']:.4f}"
              f"   item MAE {ra['individual_level_metrics']['respondent_item_mae_mean']:.4f}"
              f" vs {rb['individual_level_metrics']['respondent_item_mae_mean']:.4f}")

    print("\n=== A5: computational cost across the archive ===")
    print(f"  runs with usage records : {len(cost)}")
    print(f"  API calls               : {cost['calls'].sum():,}")
    print(f"  prompt tokens           : {cost['prompt_tokens'].sum():,}")
    print(f"  completion tokens       : {cost['completion_tokens'].sum():,}")
    print(f"  total tokens            : {(cost['prompt_tokens'] + cost['completion_tokens']).sum():,}")
    print(f"  billed cost (USD)       : ${cost['cost_usd'].sum():,.2f}")
    print(f"  median run: {cost['calls'].median():.0f} calls, "
          f"{cost['wallclock_minutes'].median():.1f} min wall-clock")
    print(f"  total wall-clock across runs: {cost['wallclock_minutes'].sum() / 60:.1f} h")
    print(f"\nWrote {args.variance_out}\nWrote {args.cost_out}")


if __name__ == "__main__":
    main()
