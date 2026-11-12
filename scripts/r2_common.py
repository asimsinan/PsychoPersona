"""Shared loaders for the round-2 offline analyses.

Every analysis works from the same per-respondent table: one row per held-out respondent
for each dataset, model family and arm, holding that respondent's metadata, human and
synthetic answers, and the errors between them. Two sources are kept apart.

``paper_split``      the split the manuscript's main tables report (seed 42), validation
                     and final test pooled. The two are disjoint, so each respondent
                     appears once.
``repeated_splits``  the ten repeated splits (seeds 101-110). A respondent can fall in
                     the held-out set of several splits; their records are averaged so
                     that the unit of analysis stays the respondent.

The deterministic profile-center baseline is added as a reference row under both
sources, computed from the same training split and profile assignment as the LLM runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs" / "experiment"
ANALYSIS = REPO / "results" / "analysis"

SCALES = {
    "anket_tr": {"digital": "digital_addiction_mean", "social": "social_anxiety_mean",
                 "wellbeing": "wellbeing_mean"},
    "bangladesh_sns": {"ucla": "ucla_loneliness_mean", "phq9": "phq9_mean", "gad7": "gad7_mean"},
}
# Demographic variables fixed before looking at any subgroup result. They are the
# respondent-context fields the prompts print, minus platform lists and usage, which are
# behavioural rather than demographic.
SUBGROUP_VARIABLES = {
    "anket_tr": ["gender", "age_band", "program", "gpa_band", "residence"],
    "bangladesh_sns": ["gender", "age_band", "education", "profession", "monthly_income", "residence"],
}
MODEL_ORDER = ["GPT-5.4", "Claude Opus 4.6", "Gemini 3.1 Pro", "Qwen3.8 27B", "DeepSeek V4.1 Flash"]
ARM_ORDER = ["DOP", "BPP", "FPNP", "BBP", "SCP", "SADP", "TPSAP", "HC"]
BASELINE = "profile-center baseline"
MIN_CELL = 15


def item_columns(ds: str, frame: pd.DataFrame) -> dict[str, list[str]]:
    """scale mean column -> its item columns, in instrument order."""
    out = {}
    for prefix, mean_col in SCALES[ds].items():
        out[mean_col] = sorted(c for c in frame.columns if c.startswith(prefix + "_item_"))
    return out


def _split_of(run: Path) -> str:
    return json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["evaluation_split"]


def load_run(run_name: str, ds: str):
    """Return (human, synthetic, train) for one run; human and train carry profile_id."""
    run = RUNS / run_name
    split = _split_of(run)
    human = pd.read_csv(run / "data" / f"{split}_split.csv")
    train = pd.read_csv(run / "data" / "train_split.csv")
    human = human.merge(pd.read_csv(run / "profiles" / f"{split}_profile_assignments.csv")
                        [["respondent_id", "profile_id"]], on="respondent_id", how="left")
    train = train.merge(pd.read_csv(run / "profiles" / "train_profile_assignments.csv")
                        [["respondent_id", "profile_id"]], on="respondent_id", how="left")
    synthetic = pd.read_csv(run / "simulation" / "simulated_responses.csv")
    synthetic = synthetic.set_index("respondent_id").loc[human.respondent_id].reset_index()
    return human, synthetic, train


def profile_center(human: pd.DataFrame, train: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """Per-profile training mean, rounded and clipped to the observed item range.

    Same rule as ``baseline_profile_center`` in ``compute_nonllm_baselines.py``.
    """
    centers = train.groupby("profile_id")[items].mean()
    lo = train[items].min().round().astype(int)
    hi = train[items].max().round().astype(int)
    rows = centers.loc[human.profile_id.values].round().clip(lower=lo, upper=hi, axis=1)
    rows.index = human.index
    out = human[["respondent_id"]].copy()
    out[items] = rows[items].values
    return out


def _errors(ds: str, human: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    scale_items = item_columns(ds, human)
    items = [c for cols in scale_items.values() for c in cols]
    h = human[items].astype(float).values
    s = synthetic[items].astype(float).values
    out = human[["respondent_id"] + SUBGROUP_VARIABLES[ds]].copy()
    out["item_mae"] = np.nanmean(np.abs(s - h), axis=1)
    for mean_col, cols in scale_items.items():
        hm = human[cols].astype(float).mean(axis=1)
        sm = synthetic[cols].astype(float).mean(axis=1)
        out[f"human__{mean_col}"] = hm.values
        out[f"synthetic__{mean_col}"] = sm.values
    return out


def _collapse(frames: list[pd.DataFrame], ds: str) -> pd.DataFrame:
    """Average a respondent's records across the splits in which they were held out."""
    allf = pd.concat(frames, ignore_index=True)
    num = [c for c in allf.columns if c == "item_mae" or "__" in c]
    meta = allf.groupby("respondent_id")[SUBGROUP_VARIABLES[ds]].first()
    means = allf.groupby("respondent_id")[num].mean()
    means["n_splits"] = allf.groupby("respondent_id").size()
    return meta.join(means).reset_index()


def respondent_table(ds: str, source: str) -> pd.DataFrame:
    """Long table: one row per respondent, model and arm, plus the baseline reference."""
    if source == "paper_split":
        idx = pd.read_csv(ANALYSIS / "run_index.csv")
        idx = idx[idx.dataset == ds].assign(seed=42)
    elif source == "repeated_splits":
        idx = pd.read_csv(ANALYSIS / "c2_llm_runs.csv")
        idx = idx[idx.dataset == ds]
    else:
        raise ValueError(source)

    out = []
    baseline_frames: dict[tuple, pd.DataFrame] = {}
    for (model, arm), cell in idx.groupby(["model", "arm"]):
        frames = []
        for _, row in cell.iterrows():
            human, synthetic, train = load_run(row.run, ds)
            frames.append(_errors(ds, human, synthetic))
            key = (row.seed, _split_of(RUNS / row.run))
            if key not in baseline_frames:
                items = [c for cols in item_columns(ds, human).values() for c in cols]
                baseline_frames[key] = _errors(ds, human, profile_center(human, train, items))
        tab = _collapse(frames, ds)
        tab.insert(0, "arm", arm)
        tab.insert(0, "model", model)
        out.append(tab)
    base = _collapse(list(baseline_frames.values()), ds)
    base.insert(0, "arm", BASELINE)
    base.insert(0, "model", BASELINE)
    out.append(base)
    full = pd.concat(out, ignore_index=True)
    full.insert(0, "source", source)
    full.insert(0, "dataset", ds)
    return full


def merge_small_levels(series: pd.Series, min_cell: int = MIN_CELL) -> pd.Series:
    """Fold levels with fewer than ``min_cell`` respondents into 'Other'."""
    series = series.fillna("Other").astype(str)
    counts = series.value_counts()
    small = counts[counts < min_cell].index
    return series.where(~series.isin(small), "Other")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = np.nanmean(values[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
