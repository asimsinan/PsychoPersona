"""Subgroup fidelity of the synthetic respondents (Reviewer 1, comment 5).

Aggregate agreement can hide systematic synthetic-human discrepancies between
demographic groups. For every dataset, model family and arm this script reports, by
gender, age band, education and the other demographic fields the prompts print:

1. item MAE per subgroup, with a bootstrap interval over respondents;
2. the spread of that error across the levels of a variable, set against what random
   relabelling of respondents produces, because the range of several noisy subgroup
   means is positive even when the subgroups do not differ;
3. signed bias of each scale mean per subgroup (synthetic minus human);
4. whether the differences between subgroups that exist in the human data are
   reproduced in the synthetic data.

The deterministic profile-center baseline goes through the same analysis. A subgroup
that is harder to approximate under a per-profile mean is harder because of the data,
not because of the language model, so the baseline is the reference for every reading.

The subgroup variables and the minimum cell size were fixed before any subgroup result
was looked at (``r2_common.SUBGROUP_VARIABLES``, ``MIN_CELL``). The number of cells is
large, so the outputs report intervals and a count of cells beyond the permutation
bound next to the count expected by chance, and no cell is singled out as significant.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_subgroup_fidelity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import r2_common as C

REPS = 2000
SEED = 20261112


def _group_means(values: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """Group means for each row of ``values`` (reps x n) given a one-hot (n x levels)."""
    return (values @ onehot) / onehot.sum(axis=0)


def analyse(cell: pd.DataFrame, ds: str, rng: np.random.Generator):
    """Return (subgroup rows, disparity rows, structure points) for one model-arm cell."""
    scales = list(C.SCALES[ds].values())
    mae = cell.item_mae.values
    sub_rows, disp_rows, points = [], [], []
    for var in C.SUBGROUP_VARIABLES[ds]:
        levels = C.merge_small_levels(cell[var])
        keep = levels.value_counts()
        keep = keep[keep >= C.MIN_CELL].index.tolist()
        if len(keep) < 2:
            continue
        mask = levels.isin(keep).values
        lv = levels[mask].values
        onehot = np.stack([(lv == k) for k in keep], axis=1).astype(float)

        for j, k in enumerate(keep):
            sel = cell[mask][lv == k]
            lo, hi = C.bootstrap_ci(sel.item_mae.values, rng, REPS)
            row = {"variable": var, "level": k, "n": len(sel),
                   "item_mae": sel.item_mae.mean(), "item_mae_lo": lo, "item_mae_hi": hi}
            for s in scales:
                bias = (sel[f"synthetic__{s}"] - sel[f"human__{s}"]).values
                blo, bhi = C.bootstrap_ci(bias, rng, REPS)
                row[f"bias__{s}"] = bias.mean()
                row[f"bias_lo__{s}"] = blo
                row[f"bias_hi__{s}"] = bhi
                # deviation of the subgroup mean from the mean of the analysed sample
                points.append({"variable": var, "level": k, "scale": s,
                               "human_dev": sel[f"human__{s}"].mean() - cell[mask][f"human__{s}"].mean(),
                               "synthetic_dev": sel[f"synthetic__{s}"].mean() - cell[mask][f"synthetic__{s}"].mean()})
            sub_rows.append(row)

        v = mae[mask]
        observed = _group_means(v[None, :], onehot)[0]
        spread = observed.max() - observed.min()
        perm = np.stack([rng.permutation(v) for _ in range(REPS)])
        null = _group_means(perm, onehot)
        null_spread = null.max(axis=1) - null.min(axis=1)
        disp = {"variable": var, "levels": len(keep), "n": int(mask.sum()),
                "spread": spread, "null_mean": null_spread.mean(),
                "null_p95": np.percentile(null_spread, 95),
                "beyond_null": bool(spread > np.percentile(null_spread, 95)),
                "highest": keep[int(observed.argmax())], "lowest": keep[int(observed.argmin())]}
        disp_rows.append(disp)
    return sub_rows, disp_rows, points


def main() -> None:
    # one row per held-out respondent, model family and arm, built from the archived runs
    table = pd.concat([C.respondent_table(ds, source) for ds in C.SCALES for source in ("paper_split", "repeated_splits")],
                      ignore_index=True)
    table.to_csv(C.ANALYSIS / "r2_respondent_errors.csv", index=False)
    rng = np.random.default_rng(SEED)
    subs, disps, structs = [], [], []
    for (ds, source, model, arm), cell in table.groupby(["dataset", "source", "model", "arm"]):
        key = {"dataset": ds, "source": source, "model": model, "arm": arm}
        s, d, pts = analyse(cell.reset_index(drop=True), ds, rng)
        subs += [{**key, **r} for r in s]
        disps += [{**key, **r} for r in d]
        if pts:
            p = pd.DataFrame(pts)
            slope = np.polyfit(p.human_dev, p.synthetic_dev, 1)[0]
            structs.append({**key, "points": len(p),
                            "correlation": np.corrcoef(p.human_dev, p.synthetic_dev)[0, 1],
                            "slope": slope,
                            "rmse": float(np.sqrt(((p.synthetic_dev - p.human_dev) ** 2).mean())),
                            "human_sd": p.human_dev.std(ddof=0)})
    sub = pd.DataFrame(subs)
    disp = pd.DataFrame(disps)
    struct = pd.DataFrame(structs)

    # how far each LLM cell's spread exceeds the baseline's spread on the same variable
    base = disp[disp.arm == C.BASELINE].set_index(["dataset", "source", "variable"]).spread
    disp["baseline_spread"] = [base.get((r.dataset, r.source, r.variable), np.nan) for r in disp.itertuples()]
    disp["excess_over_baseline"] = disp.spread - disp.baseline_spread

    sub.round(6).to_csv(C.ANALYSIS / "r2_subgroup_mae.csv", index=False)
    disp.round(6).to_csv(C.ANALYSIS / "r2_subgroup_disparity.csv", index=False)
    struct.round(6).to_csv(C.ANALYSIS / "r2_subgroup_structure.csv", index=False)

    pd.set_option("display.width", 220)
    llm = disp[disp.arm != C.BASELINE]
    print("=== spread of item MAE across subgroups, against random relabelling ===")
    for (ds, source), g in llm.groupby(["dataset", "source"]):
        b = disp[(disp.arm == C.BASELINE) & (disp.dataset == ds) & (disp.source == source)]
        print(f"{ds:15s} {source:16s} LLM cells {len(g):4d} | beyond the 95% bound {int(g.beyond_null.sum()):3d} "
              f"(expected by chance {0.05 * len(g):.0f}) | baseline cells beyond {int(b.beyond_null.sum())} of {len(b)}")
    print("\n=== by variable: share of LLM cells beyond the bound, and the same test on the baseline ===")
    by = llm.groupby(["dataset", "source", "variable"]).agg(cells=("beyond_null", "size"), beyond=("beyond_null", "sum"),
                                                           mean_spread=("spread", "mean"), null_mean=("null_mean", "mean"),
                                                           baseline_spread=("baseline_spread", "first"))
    by["baseline_beyond"] = [bool(disp[(disp.arm == C.BASELINE) & (disp.dataset == i[0]) & (disp.source == i[1])
                                       & (disp.variable == i[2])].beyond_null.any()) for i in by.index]
    print(by.round(3).to_string())
    print("\n=== are human subgroup differences reproduced? (deviation of each subgroup's scale mean) ===")
    show = struct[struct.arm.isin(["DOP", "BPP", "SADP", "HC", C.BASELINE])]
    piv = show.pivot_table(index=["dataset", "source", "model"], columns="arm", values="correlation")
    print("correlation between human and synthetic subgroup deviations:")
    print(piv.round(2).to_string())
    print(f"\nwrote r2_subgroup_mae.csv ({len(sub)} rows), r2_subgroup_disparity.csv ({len(disp)}), "
          f"r2_subgroup_structure.csv ({len(struct)})")


if __name__ == "__main__":
    main()
