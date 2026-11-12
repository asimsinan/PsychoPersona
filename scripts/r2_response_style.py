"""Response style of the synthetic respondents, and the human noise ceiling.

Two short analyses that share the item-level records.

Response style (Reviewer 5, comment 6). Work on survey responses from language models
reports a preference for the first listed option and near-uniform answer entropy
(Dominguez-Olmedo et al., 2024). This script tests both on the benchmark's records. For
every dataset, model family and arm it compares synthetic with human answers on the
share of each response category, the lowest and highest category, the scale midpoint,
the entropy of each item's answer distribution, the spread of a respondent's answers,
and straight-lining within a scale.

Human noise ceiling (Reviewer 5, comment 2). Two real people described by the same
profile do not answer alike. The item MAE between a held-out respondent and the training
respondents of the same profile, and of any profile, gives the error a real person would
score against another, which is the reference every synthetic error should be read
against.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_response_style.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as C


def _entropy(values: np.ndarray, categories: np.ndarray) -> float:
    """Mean over items of the answer entropy, as a share of the maximum (uniform)."""
    out = []
    for j in range(values.shape[1]):
        col = values[:, j]
        col = col[~np.isnan(col)]
        p = np.array([(col == k).mean() for k in categories])
        p = p[p > 0]
        out.append(-(p * np.log(p)).sum() / np.log(len(categories)))
    return float(np.mean(out))


def style(values: np.ndarray, scale_slices: list[slice], categories: np.ndarray) -> dict:
    flat = values[~np.isnan(values)]
    out = {"mean": flat.mean(),
           "share_lowest": (flat == categories[0]).mean(),
           "share_highest": (flat == categories[-1]).mean(),
           "entropy": _entropy(values, categories),
           "within_person_sd": float(np.nanmean(np.nanstd(values, axis=1))),
           "straightlining": float(np.mean([[np.nanstd(values[i, s]) == 0 for s in scale_slices]
                                            for i in range(len(values))]))}
    if len(categories) % 2 == 1:
        out["share_midpoint"] = (flat == categories[len(categories) // 2]).mean()
    for k in categories:
        out[f"share_cat_{int(k)}"] = (flat == k).mean()
    return out


def main() -> None:
    rows, ceiling = [], []
    for ds in C.SCALES:
        for source in ("paper_split", "repeated_splits"):
            if source == "paper_split":
                idx = pd.read_csv(C.ANALYSIS / "run_index.csv")
                idx = idx[idx.dataset == ds].assign(seed=42)
            else:
                idx = pd.read_csv(C.ANALYSIS / "c2_llm_runs.csv")
                idx = idx[idx.dataset == ds]
            seen: set = set()
            human_blocks = []
            for (model, arm), cell in idx.groupby(["model", "arm"]):
                syn_blocks = []
                for _, r in cell.iterrows():
                    human, synthetic, train = C.load_run(r.run, ds)
                    scale_items = C.item_columns(ds, human)
                    items = [c for cols in scale_items.values() for c in cols]
                    syn_blocks.append(synthetic[items].astype(float).values)
                    key = (r.seed, C._split_of(C.RUNS / r.run))
                    if key not in seen:
                        seen.add(key)
                        h = human[items].astype(float).values
                        human_blocks.append(h)
                        t = train[items].astype(float)
                        # error a real respondent scores against other real respondents
                        same, anyone = [], []
                        tv = t.values
                        for i in range(len(human)):
                            d = np.nanmean(np.abs(tv - h[i]), axis=1)
                            anyone.append(d.mean())
                            same.append(d[(train.profile_id == human.profile_id.iloc[i]).values].mean())
                        ceiling.append({"dataset": ds, "source": source, "seed": r.seed, "split": key[1],
                                        "n": len(human), "human_vs_same_profile": np.mean(same),
                                        "human_vs_any_respondent": np.mean(anyone)})
                # category set and scale boundaries, in the column order used above
                cats = np.arange(np.nanmin(train[items].values), np.nanmax(train[items].values) + 1)
                bounds, start = [], 0
                for cols in scale_items.values():
                    bounds.append(slice(start, start + len(cols)))
                    start += len(cols)
                rows.append({"dataset": ds, "source": source, "model": model, "arm": arm, "who": "synthetic",
                             "records": sum(len(b) for b in syn_blocks),
                             **style(np.vstack(syn_blocks), bounds, cats)})
            rows.append({"dataset": ds, "source": source, "model": "human", "arm": "human", "who": "human",
                         "records": sum(len(b) for b in human_blocks),
                         **style(np.vstack(human_blocks), bounds, cats)})

    out = pd.DataFrame(rows)
    out.round(6).to_csv(C.ANALYSIS / "r2_response_style.csv", index=False)
    ceil = pd.DataFrame(ceiling)
    ceil.round(6).to_csv(C.ANALYSIS / "r2_noise_ceiling.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["share_lowest", "share_highest", "share_midpoint", "entropy", "within_person_sd", "straightlining"]
    print("=== response style, paper split (validation and final test pooled), mean over model families ===")
    p = out[out.source == "paper_split"]
    for ds in C.SCALES:
        g = p[p.dataset == ds]
        tab = g[g.who == "synthetic"].groupby("arm")[cols].mean().reindex(C.ARM_ORDER)
        tab.loc["human"] = g[g.who == "human"][cols].iloc[0]
        print(f"-- {ds}"); print(tab.round(3).to_string())
    print("\n=== lowest-category share by model family, SADP and DOP (human in the last column) ===")
    for ds in C.SCALES:
        g = p[(p.dataset == ds) & p.arm.isin(["DOP", "SADP"])]
        tab = g.pivot(index="model", columns="arm", values="share_lowest").reindex(C.MODEL_ORDER)
        tab["human"] = p[(p.dataset == ds) & (p.who == "human")].share_lowest.iloc[0]
        print(f"-- {ds}"); print(tab.round(3).to_string())
    print("\n=== human noise ceiling: item MAE of a held-out respondent against real respondents ===")
    print(ceil.groupby(["dataset", "source"])[["human_vs_same_profile", "human_vs_any_respondent"]].mean().round(3).to_string())
    print("\nwrote r2_response_style.csv, r2_noise_ceiling.csv")


if __name__ == "__main__":
    main()
