"""Psychometric properties of the synthetic records (Reviewer 1, comment 1).

The manuscript's evidence covered distributions, correlations, scale scores and item
errors. This script adds the properties a psychometric reading asks for, each compared
between the human answers of a set of respondents and the synthetic answers generated
for those same respondents:

* reliability per scale: Cronbach's alpha, single-factor McDonald's omega and the mean
  inter-item correlation, with the estimators that produced the manuscript's Table 2;
* item discrimination in the classical sense: corrected item-rest correlations;
* category thresholds: the probit of each item's cumulative answer proportions;
* differential item functioning: for each item, an ordinal logistic regression of the
  answer on the rest score, then on group (human or synthetic), then on their
  interaction. The gain in Nagelkerke's R2 from the two group terms is the size of the
  item's differential functioning, with the rest score held equal. Nagelkerke's R2 is
  used because the published classification bands are defined on it: 0.13 and 0.26
  (Zumbo and Thomas, 1997) and 0.035 and 0.070 (Jodoin and Gierl, 2001). Both sets come
  from the literature on dichotomous items, so the same test is also run on random
  halves of the human respondents, which gives the gain to expect when no differential
  functioning exists.

Two sources are analysed and never mixed. ``repeated_splits`` pools the ten repeated
splits and keeps one synthetic record per respondent (the lowest seed in which that
respondent was held out), which gives 439 Turkish and 523 Bangladesh respondents per
arm. ``paper_split`` pools the validation and final-test respondents of the split the
manuscript reports (135 and 159). The latent-variable analyses in
``r2_invariance.R`` read the item-level records this script writes.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_psychometric_properties.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel

import r2_common as C
from compute_psychometrics import mcdonald_omega, mean_interitem_corr
from saykopersona.data import cronbach_alpha

warnings.filterwarnings("ignore")
CONTROL_REPS = 20


def item_records(ds: str, source: str) -> pd.DataFrame:
    """One synthetic record per respondent, model and arm, plus the human records."""
    if source == "paper_split":
        idx = pd.read_csv(C.ANALYSIS / "run_index.csv")
        idx = idx[idx.dataset == ds].assign(seed=42)
    else:
        idx = pd.read_csv(C.ANALYSIS / "c2_llm_runs.csv")
        idx = idx[idx.dataset == ds]
    rows, humans = [], {}
    for (model, arm), cell in idx.groupby(["model", "arm"]):
        taken: set = set()
        for _, r in cell.sort_values(["seed", "run"]).iterrows():
            human, synthetic, _ = C.load_run(r.run, ds)
            items = [c for cols in C.item_columns(ds, human).values() for c in cols]
            new = ~human.respondent_id.isin(taken).values
            part = synthetic.loc[new, ["respondent_id"] + items].copy()
            part.insert(1, "seed", r.seed)
            part.insert(0, "arm", arm)
            part.insert(0, "model", model)
            rows.append(part)
            for rid, vals in zip(human.respondent_id[new], human.loc[new, items].values):
                humans.setdefault(rid, vals)
            taken |= set(human.respondent_id[new])
    h = pd.DataFrame.from_dict(humans, orient="index", columns=items)
    h.index.name = "respondent_id"
    h = h.reset_index()
    h.insert(1, "seed", 0)
    h.insert(0, "arm", "human")
    h.insert(0, "model", "human")
    out = pd.concat([h] + rows, ignore_index=True)
    out.insert(0, "source", source)
    return out


def item_rest(frame: pd.DataFrame) -> pd.Series:
    out = {}
    for c in frame.columns:
        rest = frame.drop(columns=c).mean(axis=1)
        out[c] = np.nan if frame[c].std(ddof=1) == 0 else frame[c].corr(rest)
    return pd.Series(out)


def thresholds(col: pd.Series, cats: np.ndarray) -> np.ndarray:
    v = col.dropna().values
    cum = np.array([(v <= k).mean() for k in cats[:-1]])
    return np.where((cum > 0) & (cum < 1), norm.ppf(np.clip(cum, 1e-9, 1 - 1e-9)), np.nan)


def _fit(y: np.ndarray, X: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (Nagelkerke R2, McFadden R2, parameters) of a proportional-odds model."""
    res = OrderedModel(y, X, distr="logit").fit(method="bfgs", maxiter=200, disp=False)
    _, counts = np.unique(y, return_counts=True)
    ll_null = float((counts * np.log(counts / counts.sum())).sum())
    n = len(y)
    nagelkerke = (1.0 - np.exp(2.0 * (ll_null - res.llf) / n)) / (1.0 - np.exp(2.0 * ll_null / n))
    return float(nagelkerke), 1.0 - res.llf / ll_null, np.asarray(res.params)


def dif(y_h: pd.Series, rest_h: pd.Series, y_s: pd.Series, rest_s: pd.Series) -> dict:
    """Uniform and total DIF of one item, human against synthetic."""
    y = np.concatenate([y_h.values, y_s.values])
    rest = np.concatenate([rest_h.values, rest_s.values])
    group = np.concatenate([np.zeros(len(y_h)), np.ones(len(y_s))])
    ok = ~(np.isnan(y) | np.isnan(rest))
    y, rest, group = y[ok].astype(int), rest[ok], group[ok]
    if len(np.unique(y[group == 1])) < 2 or len(np.unique(y[group == 0])) < 2:
        return {"dif_ok": False}
    z = (rest - rest.mean()) / rest.std(ddof=1)
    try:
        n1, m1, _ = _fit(y, np.c_[z])
        n2, _, p2 = _fit(y, np.c_[z, group])
        n3, m3, _ = _fit(y, np.c_[z, group, z * group])
    except Exception:
        return {"dif_ok": False}
    return {"dif_ok": True, "dif_group_logit": float(p2[1]),
            "dif_r2_uniform": n2 - n1, "dif_r2_total": n3 - n1, "dif_mcfadden_total": m3 - m1}


def main() -> None:
    import sys
    reuse = "--reuse" in sys.argv      # rebuild the summary from the stored item table
    rel_rows, item_rows, control_rows = [], [], []
    for ds in ([] if reuse else C.SCALES):
        records = pd.concat([item_records(ds, s) for s in ("repeated_splits", "paper_split")], ignore_index=True)
        records.to_csv(C.ANALYSIS / f"r2_item_records_{ds}.csv", index=False)
        scale_items = C.item_columns(ds, records)
        all_items = [c for cols in scale_items.values() for c in cols]
        cats = np.arange(int(np.nanmin(records[all_items].values)), int(np.nanmax(records[all_items].values)) + 1)
        for source, src in records.groupby("source"):
            human = src[src.model == "human"].set_index("respondent_id")
            for (model, arm), cell in src[src.model != "human"].groupby(["model", "arm"]):
                syn = cell.set_index("respondent_id")
                hum = human.loc[syn.index]
                key = {"dataset": ds, "source": source, "model": model, "arm": arm, "n": len(syn)}
                for scale, cols in scale_items.items():
                    H, S = hum[cols].astype(float), syn[cols].astype(float)
                    rel_rows.append({**key, "scale": scale,
                                     "alpha_human": cronbach_alpha(H), "alpha_synthetic": cronbach_alpha(S),
                                     "omega_human": mcdonald_omega(H), "omega_synthetic": mcdonald_omega(S),
                                     "interitem_r_human": mean_interitem_corr(H),
                                     "interitem_r_synthetic": mean_interitem_corr(S),
                                     "zero_variance_items": int((S.std(ddof=1) == 0).sum())})
                    ir_h, ir_s = item_rest(H), item_rest(S)
                    for c in cols:
                        th, ts = thresholds(H[c], cats), thresholds(S[c], cats)
                        both = ~np.isnan(th) & ~np.isnan(ts)
                        row = {**key, "scale": scale, "item": c,
                               "item_rest_human": ir_h[c], "item_rest_synthetic": ir_s[c],
                               "categories_human": int(H[c].nunique()), "categories_synthetic": int(S[c].nunique()),
                               "threshold_mad": float(np.abs(th[both] - ts[both]).mean()) if both.any() else np.nan,
                               "thresholds_compared": int(both.sum())}
                        if source == "repeated_splits":
                            row.update(dif(H[c], H[cols].drop(columns=c).mean(axis=1),
                                           S[c], S[cols].drop(columns=c).mean(axis=1)))
                        item_rows.append(row)
            if source == "repeated_splits":
                # control: random halves of the same human respondents have no DIF
                rng = np.random.default_rng(20261112)
                for rep in range(CONTROL_REPS):
                    half = rng.permutation(len(human)) < len(human) // 2
                    a, b = human[half], human[~half]
                    for scale, cols in scale_items.items():
                        for c in cols:
                            res = dif(a[c].astype(float), a[cols].astype(float).drop(columns=c).mean(axis=1),
                                      b[c].astype(float), b[cols].astype(float).drop(columns=c).mean(axis=1))
                            control_rows.append({"dataset": ds, "rep": rep, "scale": scale, "item": c, **res})
            print(f"{ds} {source}: done", flush=True)

    if reuse:
        rel = pd.read_csv(C.ANALYSIS / "r2_psych_reliability.csv")
        items = pd.read_csv(C.ANALYSIS / "r2_psych_items.csv")
    else:
        rel = pd.DataFrame(rel_rows)
        items = pd.DataFrame(item_rows)
        rel.round(6).to_csv(C.ANALYSIS / "r2_psych_reliability.csv", index=False)
        items.round(6).to_csv(C.ANALYSIS / "r2_psych_items.csv", index=False)
        pd.DataFrame(control_rows).round(6).to_csv(C.ANALYSIS / "r2_psych_dif_control.csv", index=False)
    control = pd.read_csv(C.ANALYSIS / "r2_psych_dif_control.csv")
    bound = control[control.dif_ok == True].groupby("dataset").dif_r2_total.quantile(0.95)  # noqa: E712

    def summarise(g: pd.DataFrame) -> dict:
        valid = g.dropna(subset=["item_rest_human", "item_rest_synthetic"])
        out = {"items": len(g),
               "item_rest_mean_human": g.item_rest_human.mean(),
               "item_rest_mean_synthetic": g.item_rest_synthetic.mean(),
               "item_rest_profile_r": valid.item_rest_human.corr(valid.item_rest_synthetic) if len(valid) > 2 else np.nan,
               "items_missing_a_category": int((g.categories_synthetic < g.categories_human).sum()),
               "threshold_mad": g.threshold_mad.mean()}
        if "dif_ok" in g and g.dif_ok.notna().any():
            d = g[g.dif_ok == True]  # noqa: E712
            out.update(dif_items_tested=len(d), dif_r2_total_median=d.dif_r2_total.median(),
                       dif_share_beyond_human_halves=float((d.dif_r2_total > bound[g.dataset.iloc[0]]).mean()),
                       dif_share_jg_moderate=float((d.dif_r2_total >= 0.035).mean()),
                       dif_share_jg_large=float((d.dif_r2_total >= 0.070).mean()),
                       dif_share_zt_moderate=float((d.dif_r2_total >= 0.13).mean()),
                       dif_share_zt_large=float((d.dif_r2_total >= 0.26).mean()),
                       dif_group_logit_mean_abs=d.dif_group_logit.abs().mean())
        return out

    summary = pd.DataFrame([{"dataset": k[0], "source": k[1], "model": k[2], "arm": k[3], **summarise(g)}
                            for k, g in items.groupby(["dataset", "source", "model", "arm"])])
    summary.round(6).to_csv(C.ANALYSIS / "r2_psych_summary.csv", index=False)

    pd.set_option("display.width", 220)
    rs = rel[rel.source == "repeated_splits"]
    print("\n=== reliability, repeated splits: alpha, human against synthetic (mean over scales and model families) ===")
    t = rs.groupby(["dataset", "arm"])[["alpha_human", "alpha_synthetic", "omega_human", "omega_synthetic",
                                         "interitem_r_human", "interitem_r_synthetic"]].mean()
    print(t.round(3).to_string())
    ss = summary[summary.source == "repeated_splits"]
    print("\n=== item level, repeated splits (mean over model families) ===")
    cols = ["item_rest_mean_human", "item_rest_mean_synthetic", "item_rest_profile_r", "items_missing_a_category",
            "threshold_mad"]
    print(ss.groupby(["dataset", "arm"])[cols].mean().round(3).to_string())
    print("\n=== differential item functioning, Nagelkerke R2 gain (mean over model families) ===")
    print("human random halves, 95th percentile of the gain:", bound.round(4).to_dict())
    dcols = ["dif_r2_total_median", "dif_share_beyond_human_halves", "dif_share_jg_moderate",
             "dif_share_jg_large", "dif_share_zt_moderate", "dif_share_zt_large"]
    print(ss.groupby(["dataset", "arm"])[dcols].mean().round(3).to_string())
    print("\nwrote r2_item_records_<dataset>.csv, r2_psych_reliability.csv, r2_psych_items.csv, r2_psych_summary.csv")


if __name__ == "__main__":
    main()
