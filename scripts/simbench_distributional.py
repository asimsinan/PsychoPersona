"""SimBench-style group-level distributional scoring (Reviewer 1, comment 3).

Reviewer 1 notes that the benchmark does not reproduce a specific method from the recent
simulation literature. SimBench evaluates simulated respondents by comparing the
*distribution* of answers against the human distribution at group level, rather than by
matching individual respondents. That target is computable directly on the records this
benchmark already generates, so it is implemented here as an additional scoring pass over
existing runs -- no new generation is required.

Metrics, all computed per item and then averaged, and all lower-is-better except where
noted:

``TVD``      Total variation distance between the human and simulated response
             distributions for an item, i.e. half the L1 distance over the response
             categories. Bounded in [0, 1]; 0 means identical distributions.
``JSD``      Jensen-Shannon divergence in bits, a smoothed symmetric alternative to KL
             that stays finite when a category is unused by one side.
``WD``       1-Wasserstein (earth-mover) distance on the ordinal response scale,
             normalised by the scale width so it is comparable across the 1-5 and 0-3
             instruments. Unlike TVD and JSD this respects response *ordering*, so
             confusing adjacent categories costs less than confusing extreme ones.
``CoverageR`` Share of human response categories that the simulated data uses at all.
             Higher is better; it detects the mode collapse that distribution-level
             evaluations are meant to catch.

A method can score well here while approximating no individual respondent, which is the
point: this is the group-level validity target, reported alongside the respondent-level
metrics rather than instead of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import saykopersona.dataset_catalog as catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs" / "experiment"
ARMS = {
    "demographic_only_v1": "DOP", "v3_tr_gpt_b": "BPP", "v3_tr_gpt_d": "FPNP",
    "v3_tr_gpt_e": "BBP", "direct_stats_v1": "SADP",
    "direct_stats_two_pass_v1": "TPSAP", "v4_tr_hybrid_b": "HC",
}
MODELS = {"openai/gpt-5.4": "GPT-5.4", "anthropic/claude-opus-4.6": "Claude Opus 4.6",
          "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
          "qwen/qwen3.8-27b": "Qwen3.8 27B",
          "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash"}


def _settings(name: str):
    return (catalog.default_dataset_settings() if name == "anket_tr"
            else catalog.bangladesh_sns_dataset_settings())


def _hist(values: pd.Series, lo: int, hi: int) -> np.ndarray:
    counts = np.zeros(hi - lo + 1, dtype=float)
    clean = values.dropna().astype(int)
    for v in clean:
        if lo <= v <= hi:
            counts[v - lo] += 1
    total = counts.sum()
    return counts / total if total else counts


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _wasserstein(p: np.ndarray, q: np.ndarray, width: int) -> float:
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum()) / width if width else float("nan")


def score_run(run_dir: Path) -> dict | None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    sim_path = run_dir / "simulation" / "simulated_responses.csv"
    split = manifest.get("evaluation_split")
    human_path = run_dir / "data" / f"{split}_split.csv"
    if not (sim_path.exists() and human_path.exists()):
        return None

    cfg = manifest["config"]
    ds = cfg.get("dataset")
    ds = ds.get("dataset_id") if isinstance(ds, dict) else "anket_tr"
    settings = _settings(ds)
    items = [i for s in settings.scales for i in s.item_columns]
    lo, hi = settings.response_min, settings.response_max

    human = pd.read_csv(human_path)
    sim = pd.read_csv(sim_path)
    common = sorted(set(human["respondent_id"]) & set(sim["respondent_id"]))
    human = human[human["respondent_id"].isin(common)]
    sim = sim[sim["respondent_id"].isin(common)]

    tvd, jsd, wd, cov = [], [], [], []
    for item in items:
        if item not in human.columns or item not in sim.columns:
            continue
        p, q = _hist(human[item], lo, hi), _hist(sim[item], lo, hi)
        if p.sum() == 0 or q.sum() == 0:
            continue
        tvd.append(_tvd(p, q))
        jsd.append(_jsd(p, q))
        wd.append(_wasserstein(p, q, hi - lo))
        used_by_human = p > 0
        cov.append(float((q[used_by_human] > 0).mean()) if used_by_human.any() else np.nan)

    scs = cfg["simulation"].get("self_consistency_samples") or 1
    pv = cfg["simulation"]["prompt_version"]
    arm = "SCP" if scs > 1 and pv == "v3_tr_gpt_b" else ARMS.get(pv, pv)
    return {
        "dataset": ds, "arm": arm,
        "model": MODELS.get(manifest.get("model_id"), manifest.get("model_id")),
        "split": split, "n": len(common), "n_items": len(tvd),
        "TVD": round(float(np.mean(tvd)), 6),
        "JSD_bits": round(float(np.mean(jsd)), 6),
        "WD_norm": round(float(np.mean(wd)), 6),
        "CoverageR": round(float(np.nanmean(cov)), 6),
        "run": run_dir.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("results/analysis/simbench_distributional.csv"))
    args = parser.parse_args()

    index = pd.read_csv(REPO_ROOT / "results" / "analysis" / "run_index.csv")
    rows = []
    for run_name in index["run"]:
        d = RUNS_ROOT / run_name
        if not (d / "run_manifest.json").exists():
            continue
        try:
            row = score_run(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {d.name}: {exc}")
            continue
        if row:
            rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print("=== C3: SimBench-style group-level distributional scoring ===")
    print("    TVD / JSD / WD lower is better; CoverageR higher is better\n")
    order = ["DOP", "BPP", "FPNP", "BBP", "SCP", "SADP", "TPSAP", "HC"]
    for ds in ["anket_tr", "bangladesh_sns"]:
        sub = frame[frame.dataset == ds]
        if sub.empty:
            continue
        for split in ["validation", "final_test"]:
            s = sub[sub.split == split]
            if s.empty:
                continue
            agg = (s.groupby("arm")[["TVD", "JSD_bits", "WD_norm", "CoverageR"]]
                   .mean().round(4).reindex([a for a in order if a in set(s.arm)]))
            print(f"--- {ds} / {split} (mean over model families) ---")
            print(agg.to_string(), "\n")
    print(f"Wrote {args.out}  ({len(frame)} runs scored)")


if __name__ == "__main__":
    main()
