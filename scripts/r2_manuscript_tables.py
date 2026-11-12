"""Tables of the round-2 manuscript, written from the analysis files.

Every table that the revision adds, extends or moves to the appendix is generated here,
so that no table cell is typed by hand. The script rewrites the blocks of
``ipm/manuscript.tex`` that sit between a pair of marker lines

    %% BEGIN GENERATED <name>
    %% END GENERATED <name>

and leaves everything else in the file untouched. A block that has no markers in the
manuscript yet is only written to ``results/analysis/tables/<name>.tex``.

Blocks
------
respondent-level   main text: SADP on the Turkish final test, five families
psychometric       main text: psychometric properties and invariance, four arms
appendix           all appendix tables

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_manuscript_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
A = REPO / "results" / "analysis"
OUT = A / "tables"
TEX = REPO / "ipm" / "manuscript.tex"
MODELS = ["GPT-5.4", "Claude Opus 4.6", "Gemini 3.1 Pro", "Qwen3.8 27B", "DeepSeek V4.1 Flash"]
DS = {"anket_tr": "Turkish", "bangladesh_sns": "Bangladesh"}
KEY_ARMS = ["DOP", "BPP", "SADP", "HC"]
ALL_ARMS = ["DOP", "BPP", "FPNP", "BBP", "SCP", "SADP", "TPSAP", "HC"]
BASELINES = [("marginal", "Marginal sampler", "none"), ("grm_population", "Graded response model", "none"),
             ("copula_population", "Gaussian copula", "none"), ("profile_center", "Profile mean", "profile"),
             ("within_profile", "Within-profile sampler", "profile"), ("grm_profile", "Graded response model", "profile"),
             ("copula_profile", "Gaussian copula", "profile"), ("nn_metadata", "Nearest neighbor", "metadata"),
             ("ordreg_metadata", "Regression, sampled", "metadata"),
             ("ordreg_metadata_median", "Regression, median", "metadata"),
             ("ordreg_profile_metadata", "Regression, sampled", "profile, metadata"),
             ("ordreg_profile_metadata_median", "Regression, median", "profile, metadata")]


def f(x, d: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def table(caption: str, label: str, spec: str, header: str, body: list[str], size: str = "scriptsize",
          note: str | None = None) -> str:
    lines = ["\\begin{table}[pos=!htbp]", f"\\{size}\\centering\\setlength{{\\tabcolsep}}{{2.5pt}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}", "\\toprule", header + " \\\\",
             "\\midrule", *body, "\\bottomrule", "\\end{tabular*}"]
    if note:
        lines += ["\\par\\smallskip", f"\\{size} {note}"]
    lines.append("\\end{table}")
    return "\n".join(lines)


def panel(title: str, ncols: int) -> str:
    return f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"


# --------------------------------------------------------------------------- #
grid = pd.read_csv(A / "r2_main_grid.csv")


def grid_rows(ds: str, split: str, arms: list[str]) -> list[str]:
    g = grid[(grid.dataset == ds) & (grid.split == split)]
    out = []
    for arm in arms:
        for assignment, label in (("oracle", "Oracle"), ("coldstart", "Cold-start")):
            if arm == "DOP" and assignment == "coldstart":
                continue
            for m in MODELS:
                r = g[(g.arm == arm) & (g.assignment == assignment) & (g.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                out.append(" & ".join([arm, m, "--" if arm == "DOP" else label, f(r.corr_rmse), f(r.within_rmse), f(r.recovery),
                                       f(r.scale_rmse), f(r.item_mae), f(r.item_emr), f(r["unique"]), f(r.stability)]) + " \\\\")
        out.append("\\addlinespace[2pt]")
    return out[:-1]


GRID_HEAD = "Method & Model & Assignment & Corr & Within & Recovery & Scale RMSE & Item MAE & Item EMR & Unique & Stability"
GRID_NOTE = ("DOP receives no profile and takes a single row. Entries marked n/d are not defined. Recovery is not "
             "defined when every held-out respondent is assigned one profile, and correlation RMSE is not defined when "
             "the generated scale means carry no variance.")


def t_grid() -> list[str]:
    maj = grid[(grid.split == "final_test") & (grid.assignment == "oracle")].groupby("dataset").majority.first()
    majc = grid[(grid.split == "final_test") & (grid.assignment == "coldstart")].groupby("dataset").majority.first()
    majv = grid[(grid.split == "validation") & (grid.assignment == "oracle") & (grid.dataset == "bangladesh_sns")].majority.iloc[0]
    return [
        table(f"Turkish final-test results for DOP, BPP, SADP and HC on the five model families. The majority-profile "
              f"rate is {f(maj['anket_tr'])} under oracle assignment and {f(majc['anket_tr'])} under cold-start assignment.",
              "tab:grid-tr", "lllcccccccc", GRID_HEAD, grid_rows("anket_tr", "final_test", KEY_ARMS), note=GRID_NOTE),
        table(f"Bangladesh final-test results for DOP, BPP, SADP and HC on the five model families. The majority-profile "
              f"rate is {f(maj['bangladesh_sns'])} under oracle assignment.",
              "tab:grid-bd", "lllcccccccc", GRID_HEAD, grid_rows("bangladesh_sns", "final_test", KEY_ARMS), note=GRID_NOTE),
        table(f"Bangladesh validation results for DOP, BPP and SADP on the five model families. The majority-profile "
              f"rate is {f(majv)} under oracle assignment.",
              "tab:grid-bd-val", "lllcccccccc", GRID_HEAD, grid_rows("bangladesh_sns", "validation", ["DOP", "BPP", "SADP"]),
              note=GRID_NOTE),
    ]


def t_respondent_level() -> str:
    g = grid[(grid.dataset == "anket_tr") & (grid.split == "final_test") & (grid.arm == "SADP")]
    dop = grid[(grid.dataset == "anket_tr") & (grid.split == "final_test") & (grid.arm == "DOP")
               & grid.model.isin(MODELS[:3])].item_mae.mean()
    body = []
    for m in MODELS:
        for assignment, label in (("oracle", "Oracle"), ("coldstart", "Cold-start")):
            r = g[(g.model == m) & (g.assignment == assignment)].iloc[0]
            body.append(" & ".join([m, "SADP", label, f(r.scale_rmse), f(r.item_mae), f(r.item_emr), f(r.full_vector_emr)]) + " \\\\")
    return table("Turkish final-test respondent-level approximation for SADP under oracle and cold-start profile "
                 f"assignment. Demographic-only prompting, which receives no profile, reaches an item MAE of {f(dop)} "
                 "averaged over the three hosted families on this split.", "tab:respondent-level", "lllcccc",
                 "Model & Method & Assignment & Scale RMSE & Item MAE & Item EMR & Full vector EMR", body, size="small")


# --------------------------------------------------------------------------- #
def t_splits() -> list[str]:
    d = pd.read_csv(A / "c2_llm_arm_dispersion.csv")
    metrics = ["Corr RMSE", "Profile recovery", "Item MAE"]
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 5))
        for arm in ALL_ARMS:
            for m in MODELS:
                r = d[(d.dataset == ds) & (d.arm == arm) & (d.model == m)].set_index("metric")
                if r.empty:
                    continue
                body.append(" & ".join([arm, m] + [f"{f(r.loc[k, 'mean'])} ({f(r.loc[k, 'sd'])})" for k in metrics]) + " \\\\")
        body.append("\\midrule")
    disp = table("Between-split dispersion of each arm across the ten repeated splits, final-test split. Each cell gives "
                 "the mean across splits with the standard deviation in parentheses.", "tab:c2-arms", "llccc",
                 "Arm & Model & Corr RMSE & Profile recovery & Item MAE", body[:-1])
    c = pd.read_csv(A / "c2_llm_contrasts.csv")
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 5))
        for con, lab in (("SADP-DOP", "SADP $-$ DOP"), ("BPP-DOP", "BPP $-$ DOP"), ("SADP-BPP", "SADP $-$ BPP"), ("HC-SADP", "HC $-$ SADP")):
            for m in MODELS:
                r = c[(c.dataset == ds) & (c.contrast == con) & (c.model == m)].set_index("metric")
                cells = [f"{f(r.loc[k, 'mean_delta'], sign=True)} ({int(round(r.loc[k, 'sign_preserved'] * r.loc[k, 'n_splits']))}/{int(r.loc[k, 'n_splits'])})"
                         for k in metrics]
                body.append(" & ".join([lab, m] + cells) + " \\\\")
        body.append("\\midrule")
    con = table("Paired contrasts across the ten repeated splits, by model family. Each cell gives the mean difference "
                "between the two arms, computed within split, followed by the number of splits in which the difference "
                "takes the sign of that mean.", "tab:c2-contrasts", "llccc",
                "Contrast & Model & Corr RMSE & Profile recovery & Item MAE", body[:-1])
    return [disp, con]


# --------------------------------------------------------------------------- #
def t_distributional() -> str:
    s = pd.read_csv(A / "simbench_distributional.csv")
    s = s[(s.split == "final_test") & s.model.isin(MODELS[:3])]
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 5))
        m = s[s.dataset == ds].groupby("arm")[["TVD", "JSD_bits", "WD_norm", "CoverageR"]].mean()
        for arm in ALL_ARMS:
            body.append(" & ".join([arm] + [f(m.loc[arm, k]) for k in m.columns]) + " \\\\")
        body.append("\\midrule")
    return table("Group-level distributional fidelity on the final-test split, averaged over items and over the three "
                 "hosted model families. TVD, JSD and WD are lower-is-better; coverage is higher-is-better.",
                 "tab:distributional", "lcccc", "Arm & TVD & JSD (bits) & Wasserstein & Coverage", body[:-1])


def t_uncertainty() -> list[str]:
    u = pd.read_csv(A / "uncertainty_summary.csv")
    names = {"ScaleRMSE": "Scale RMSE", "ItemMAE": "Item MAE", "ItemEMR": "Item EMR"}
    body = []
    for ds, name in DS.items():
        n = int(u[u.dataset == ds].n.iloc[0])
        body.append(panel(f"{name}, $n={n}$", 4))
        for arm in ["SADP", "BPP", "HC", "Profile-center"]:
            for i, (k, lab) in enumerate(names.items()):
                r = u[(u.dataset == ds) & (u.arm == arm) & (u.metric == k)].iloc[0]
                body.append(" & ".join([arm.replace("Profile-center", "Profile mean") if i == 0 else "", lab, f(r["mean"]),
                                        f"[{f(r.ci_lo)}, {f(r.ci_hi)}]"]) + " \\\\")
        body.append("\\midrule")
    arms = table("Bootstrap 95\\% confidence intervals for respondent-level means on the final-test split of each "
                 "dataset, GPT-5.4, 2000 resamples.", "tab:ci-arms", "llcc", "Arm & Metric & Mean & 95\\% CI", body[:-1])
    d = pd.read_csv(A / "uncertainty_deltas.csv")
    labs = {"SADP_vs_BPP": "SADP $-$ BPP", "SADP_vs_ProfileCenter": "SADP $-$ profile mean", "SADP_vs_HC": "SADP $-$ HC",
            "BPP_vs_HC": "BPP $-$ HC"}
    body = []
    for ds, name in DS.items():
        n = int(d[d.dataset == ds].n.iloc[0])
        body.append(panel(f"{name}, $n={n}$", 5))
        for con, lab in labs.items():
            for i, (k, mlab) in enumerate(names.items()):
                r = d[(d.dataset == ds) & (d.contrast == con) & (d.metric == k)].iloc[0]
                star = "$^{*}$" if (r.delta_ci_lo > 0) == (r.delta_ci_hi > 0) else ""
                body.append(" & ".join([lab if i == 0 else "", mlab, f(r.mean_delta, sign=True) + star,
                                        f"[{f(r.delta_ci_lo, sign=True)}, {f(r.delta_ci_hi, sign=True)}]",
                                        f(r.cohens_dz, 2, sign=True)]) + " \\\\")
        body.append("\\midrule")
    con = table("Paired contrasts on the final-test split of each dataset, GPT-5.4. Negative values favor the "
                "first-named method for the error metrics and the second-named method for item EMR. An asterisk marks an "
                "interval excluding zero.", "tab:ci-contrasts", "llccc",
                "Contrast & Metric & Mean difference & 95\\% CI & $d_z$", body[:-1])
    return [arms, con]


# --------------------------------------------------------------------------- #
def t_baselines() -> list[str]:
    b = pd.read_csv(A / "r2_baselines_paper.csv")
    tvd = pd.read_csv(A / "simbench_distributional.csv")
    tvd = tvd[(tvd.split == "final_test") & (tvd.model == "GPT-5.4")].set_index(["dataset", "arm"]).TVD
    out = []
    head = "Method & Information & Corr & Scale RMSE & Item MAE & Item EMR & Recovery & Unique & TVD & Cold-start MAE"
    for ds, name, label in (("anket_tr", "Turkish", "tab:baselines-tr"), ("bangladesh_sns", "Bangladesh", "tab:baselines-bd")):
        g = grid[(grid.dataset == ds) & (grid.split == "final_test") & (grid.model == "GPT-5.4")]
        body = [panel("LLM arms, GPT-5.4", 10)]
        for arm in ALL_ARMS:
            o = g[(g.arm == arm) & (g.assignment == "oracle")].iloc[0]
            c = g[(g.arm == arm) & (g.assignment == "coldstart")]
            info = "metadata" if arm == "DOP" else "profile, metadata"
            body.append(" & ".join([arm, info, f(o.corr_rmse), f(o.scale_rmse), f(o.item_mae), f(o.item_emr), f(o.recovery),
                                    f(o["unique"]), f(tvd[(ds, arm)]), "--" if c.empty else f(c.item_mae.iloc[0])]) + " \\\\")
        body += ["\\midrule", panel("Non-LLM baselines", 10)]
        for method, lab, info in BASELINES:
            o = b[(b.dataset == ds) & (b.method == method) & (b.setting == "oracle")].iloc[0]
            c = b[(b.dataset == ds) & (b.method == method) & (b.setting == "coldstart")].iloc[0]
            cold = f(c.RespItemMAE) if "profile" in info else "--"
            body.append(" & ".join([lab, info, f(o.CorrRMSE), f(o.RespScaleRMSE), f(o.RespItemMAE), f(o.ItemEMR), f(o.PRA),
                                    f(o.SimUniqueRatio), f(o.TVD), cold]) + " \\\\")
        maj = float(g[g.assignment == "oracle"].majority.iloc[0])
        out.append(table(f"{name} final-test results for the GPT-5.4 arms and the non-LLM baselines under oracle assignment, "
                         f"with item MAE under cold-start assignment in the last column. The majority-profile rate is {f(maj)}.",
                         label, "llcccccccc", head, body,
                         note="Sampled baselines are averaged over 50 replicates. The information column names what a method "
                              "knows about a held-out respondent. A cold-start value is given for methods that use the profile. "
                              "Regression is the proportional-odds regression, with answers sampled from the predicted "
                              "probabilities or set to the predicted median."))
    s = pd.read_csv(A / "r2_baselines_splits.csv")
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 7))
        for method, lab, info in BASELINES:
            o = s[(s.dataset == ds) & (s.method == method) & (s.setting == "oracle")]
            c = s[(s.dataset == ds) & (s.method == method) & (s.setting == "coldstart")]
            cell = lambda d, k: f"{f(d[k].mean())} ({f(d[k].std())})"  # noqa: E731
            body.append(" & ".join([lab, info, cell(o, "RespItemMAE"), cell(c, "RespItemMAE") if "profile" in info else "--",
                                    cell(o, "CorrRMSE"), cell(o, "PRA"), cell(o, "SimUniqueRatio")]) + " \\\\")
        body.append("\\midrule")
    out.append(table("Non-LLM baselines across 50 random splits with profiles refitted on every training split. Each cell "
                     "gives the mean across splits with the standard deviation in parentheses.", "tab:baselines-splits",
                     "llccccc", "Method & Information & Item MAE & Cold-start MAE & Corr RMSE & Recovery & Unique",
                     body[:-1]))
    return out


# --------------------------------------------------------------------------- #
def invariance_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset, model and arm, and for the human split-half control: rejection rates and change in CFI."""
    def summarise(fits: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        cont = fits[fits.sequence == "continuous"]
        unit = [k for k in fits.columns if k in ("dataset", "model", "arm", "scale", "rep")]
        wide = cont.pivot_table(index=unit, columns="step", values=["cfi", "p_difference"], aggfunc="first")
        w = pd.DataFrame({"metric_rejected": wide[("p_difference", "metric")] < 0.05,
                          "scalar_rejected": wide[("p_difference", "scalar")] < 0.05,
                          "dcfi_metric": wide[("cfi", "metric")] - wide[("cfi", "configural")],
                          "dcfi_scalar": wide[("cfi", "scalar")] - wide[("cfi", "metric")]}).reset_index()
        out = w.groupby(keys).agg(tests=("metric_rejected", "size"), metric_rejected=("metric_rejected", "mean"),
                                  scalar_rejected=("scalar_rejected", "mean"), dcfi_metric=("dcfi_metric", "mean"),
                                  dcfi_scalar=("dcfi_scalar", "mean"))
        ordinal = fits[(fits.sequence == "ordinal") & (fits.step == "configural")]
        out["merged_categories"] = ordinal.groupby(keys).merged_categories.mean()
        return out.reset_index()
    fits = pd.read_csv(A / "r2_invariance_fits.csv")
    control = pd.read_csv(A / "r2_invariance_control.csv")
    return summarise(fits, ["dataset", "model", "arm"]), summarise(control, ["dataset"])


def psych_frames():
    s = pd.read_csv(A / "r2_psych_summary.csv")
    s = s[s.source == "repeated_splits"]
    r = pd.read_csv(A / "r2_psych_reliability.csv")
    r = r[r.source == "repeated_splits"].groupby(["dataset", "model", "arm"])[
        ["alpha_human", "alpha_synthetic", "interitem_r_human", "interitem_r_synthetic"]].mean().reset_index()
    inv, ctrl = invariance_summary()
    full = s.merge(r, on=["dataset", "model", "arm"]).merge(inv, on=["dataset", "model", "arm"], how="left")
    return full, ctrl


def t_psychometric_main() -> str:
    full, ctrl = psych_frames()
    cols = ["alpha_synthetic", "interitem_r_synthetic", "item_rest_mean_synthetic", "items_missing_a_category",
            "dif_share_beyond_human_halves", "metric_rejected", "scalar_rejected", "dcfi_scalar"]
    body = []
    for ds, name in DS.items():
        g = full[(full.dataset == ds) & full.arm.isin(KEY_ARMS)]
        items = int(g["items"].iloc[0])
        body.append(panel(f"{name}, {items} items", 9))
        c = ctrl[ctrl.dataset == ds].iloc[0]
        body.append(" & ".join(["Human", f(g.alpha_human.iloc[0], 2), f(g.interitem_r_human.iloc[0], 2),
                                f(g.item_rest_mean_human.iloc[0], 2), "0", "0.05", f(c.metric_rejected, 2),
                                f(c.scalar_rejected, 2), f(c.dcfi_scalar, sign=True)]) + " \\\\")
        m = g.groupby("arm")[cols].mean()
        for arm in KEY_ARMS:
            v = m.loc[arm]
            body.append(" & ".join([arm, f(v.alpha_synthetic, 2), f(v.interitem_r_synthetic, 2), f(v.item_rest_mean_synthetic, 2),
                                    f(v.items_missing_a_category, 1), f(v.dif_share_beyond_human_halves, 2),
                                    f(v.metric_rejected, 2), f(v.scalar_rejected, 2), f(v.dcfi_scalar, sign=True)]) + " \\\\")
        body.append("\\midrule")
    return table("Psychometric properties of the synthetic records against the human records of the same respondents, "
                 "repeated splits, mean over the five model families.", "tab:psychometric", "lcccccccc",
                 "Records & $\\alpha$ & $\\bar{r}$ & Item-rest $r$ & Items lacking a category & DIF items & Metric rejected & "
                 "Scalar rejected & $\\Delta$CFI, scalar", body[:-1], size="scriptsize",
                 note="$\\alpha$, $\\bar{r}$ and the item-rest correlation are means over the three scales. DIF items is the "
                      "share of items whose gain in Nagelkerke $R^2$ exceeds the bound that 95\\% of items stay under when the "
                      "human respondents are split at random. Metric and scalar rejected are the shares of scale-level "
                      "difference tests rejected at 0.05. The human row gives the same quantities for random halves of the "
                      "human respondents.")


def t_psychometric_appendix() -> list[str]:
    full, ctrl = psych_frames()
    g = full[full.arm.isin(KEY_ARMS)]
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 9))
        for arm in KEY_ARMS:
            for m in MODELS:
                r = g[(g.dataset == ds) & (g.arm == arm) & (g.model == m)].iloc[0]
                body.append(" & ".join([arm, m, f(r.alpha_synthetic, 2), f(r.interitem_r_synthetic, 2), f(r.item_rest_mean_synthetic, 2),
                                        f(r.item_rest_profile_r, 2), str(int(r.items_missing_a_category)), f(r.threshold_mad, 2),
                                        f(r.dif_share_beyond_human_halves, 2)]) + " \\\\")
        body.append("\\midrule")
    classical = table("Classical item and scale properties of the synthetic records by model family, repeated splits.",
                      "tab:psychometric-families", "llccccccc",
                      "Arm & Model & $\\alpha$ & $\\bar{r}$ & Item-rest $r$ & Item-rest profile $r$ & Items lacking a category & "
                      "Threshold MAD & DIF items", body[:-1],
                      note="Item-rest profile $r$ is the correlation between the human and the synthetic item-rest correlations "
                           "across items. Threshold MAD is the mean absolute difference between human and synthetic category "
                           "thresholds in probit units.")
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 8))
        c = ctrl[ctrl.dataset == ds].iloc[0]
        body.append(" & ".join(["Human halves", "--", str(int(c.tests)), f(c.metric_rejected, 2), f(c.scalar_rejected, 2),
                                f(c.dcfi_metric, sign=True), f(c.dcfi_scalar, sign=True), f(c.merged_categories, 1)]) + " \\\\")
        for arm in KEY_ARMS:
            for m in MODELS:
                r = g[(g.dataset == ds) & (g.arm == arm) & (g.model == m)].iloc[0]
                tests = 0 if np.isnan(r.tests) else int(r.tests)
                body.append(" & ".join([arm, m, str(tests), f(r.metric_rejected, 2), f(r.scalar_rejected, 2),
                                        f(r.dcfi_metric, sign=True), f(r.dcfi_scalar, sign=True), f(r.merged_categories, 1)]) + " \\\\")
        body.append("\\midrule")
    inv = table("Measurement invariance between human and synthetic records by model family, one-factor model per scale, "
                "repeated splits.", "tab:invariance-families", "llcccccc",
                "Arm & Model & Tests & Metric rejected & Scalar rejected & $\\Delta$CFI, metric & $\\Delta$CFI, scalar & "
                "Merged categories", body[:-1],
                note="Tests is the number of scales, or of scales by random halves for the human control. Rejection is at "
                     "0.05 on the scaled difference test of the continuous sequence. Merged categories is the mean number of "
                     "response categories per scale that had to be merged before the ordinal sequence could be fitted. "
                     "A scale whose model did not converge is not counted, and n/d marks a cell with no converged model.")
    return [classical, inv]


# --------------------------------------------------------------------------- #
def t_profiles() -> list[str]:
    ag = pd.read_csv(A / "r2_profile_agreement.csv")
    bl = pd.read_csv(A / "r2_profile_baselines.csv")
    rs = pd.read_csv(A / "r2_profile_rescoring.csv")
    rs = rs[rs.same_count & rs.arm.isin(KEY_ARMS)]
    wide = rs.pivot_table(index=["dataset", "source", "seed", "model", "method"], columns="arm", values="recovery").dropna().reset_index()
    labels = {"pipeline": "Mixture, full (benchmark)", "gmm_diag": "Mixture, diagonal",
              "gmm_tied": "Mixture, tied", "kmeans": "k-means", "ward": "Ward"}
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 8))
        for method, lab in labels.items():
            a = ag[(ag.dataset == ds) & (ag.method == method) & ag.same_count]
            b = bl[(bl.dataset == ds) & (bl.method == method) & bl.same_count & (bl.baseline == "profile_center")]
            w = wide[(wide.dataset == ds) & (wide.method == method)]
            body.append(" & ".join([lab, f(a.ari_train.mean(), 2), f(b.RespItemMAE.mean()), f(b.PRA.mean()),
                                    f((w.BPP > w.DOP).mean(), 2), f((w.SADP > w.DOP).mean(), 2), f((w.SADP >= w.BPP).mean(), 2),
                                    f((w.HC < w.SADP).mean(), 2)]) + " \\\\")
        body.append("\\midrule")
    sens = table("Sensitivity to the clustering method at the profile count the benchmark selects.", "tab:profile-sensitivity",
                 "lccccccc", "Clustering method & ARI & Item MAE & Recovery & BPP $>$ DOP & "
                 "SADP $>$ DOP & SADP $\\geq$ BPP & HC $<$ SADP", body[:-1],
                 note="Full, diagonal and tied name the covariance structure of the mixture. ARI is the adjusted Rand index "
                      "against the benchmark's partition on the training split. Item MAE and recovery are those of the "
                      "profile-mean baseline under the method. These three columns are means of 50 splits. The last four columns give the share of model-by-split "
                      "cells in which the recovery contrast holds when the held-out respondents of the existing LLM runs "
                      "are relabeled by the method.")
    r = pd.read_csv(A / "r2_alt_profile_runs.csv")
    body = []
    for variant, vlab in (("kmeans", "k-means profiles, oracle assignment"), ("soft", "Soft membership, cold-start assignment")):
        body.append(panel(vlab, 9))
        for ds, name in DS.items():
            for arm in ["BPP", "SADP"]:
                for split, slab in (("validation", "Validation"), ("final_test", "Final test")):
                    x = r[(r.variant == variant) & (r.dataset == ds) & (r.arm == arm) & (r.split == split)].iloc[0]
                    body.append(" & ".join([name, arm, slab, f(x.RespItemMAE_reference), f(x.RespItemMAE_variant),
                                            f"{f(x.item_mae_difference, sign=True)} [{f(x.item_mae_difference_lo, sign=True)}, {f(x.item_mae_difference_hi, sign=True)}]",
                                            f"{f(x.PRA_reference)} / {f(x.PRA_variant)}",
                                            f"{f(x.SimUniqueRatio_reference)} / {f(x.SimUniqueRatio_variant)}",
                                            f"{f(x.center_item_mae_reference)} / {f(x.center_item_mae_variant)}"]) + " \\\\")
        body.append("\\midrule")
    alt = table("Generation under alternative profile constructions, GPT-5.4. Each run is compared with the benchmark run "
                "of the same arm and split on the same held-out respondents.", "tab:alt-profiles", "lllcccccc",
                "Dataset & Arm & Split & Benchmark & Variant & Difference [95\\% CI] & Recovery & Unique & "
                "Profile mean", body[:-1],
                note="Benchmark, variant and difference are item MAE. Profile mean is the item MAE of the profile-mean "
                     "baseline. Paired values are given as benchmark / variant. The difference is variant minus benchmark, paired by "
                     "respondent, with a bootstrap interval over respondents.")
    return [sens, alt]


def t_subgroup() -> list[str]:
    d = pd.read_csv(A / "r2_subgroup_disparity.csv")
    d = d[d.source == "paper_split"]
    names = {"gender": "Gender", "age_band": "Age band", "program": "Program", "gpa_band": "Grade band", "residence": "Residence",
             "education": "Education", "profession": "Profession", "monthly_income": "Monthly income"}
    body = []
    for ds, name in DS.items():
        body.append(panel(name, 7))
        llm = d[(d.dataset == ds) & (d.arm != "profile-center baseline")]
        base = d[(d.dataset == ds) & (d.arm == "profile-center baseline")].set_index("variable")
        for var in llm.variable.unique():
            v = llm[llm.variable == var]
            body.append(" & ".join([names.get(var, var), str(int(v.levels.iloc[0])), f"{int(v.beyond_null.sum())} of {len(v)}",
                                    f(v.spread.mean()), f(v.null_mean.mean()), f(base.loc[var, "spread"]),
                                    "yes" if bool(base.loc[var, "beyond_null"]) else "no"]) + " \\\\")
        body.append("\\midrule")
    disp = table("Spread of item MAE across the levels of each demographic variable, validation and final-test respondents "
                 "of the paper's split.", "tab:subgroup", "lcccccc",
                 "Variable & Levels & Cells beyond bound & Spread, LLM & Spread, relabeled & "
                 "Spread, profile mean & Beyond bound", body[:-1],
                 note="Spread is the largest minus the smallest subgroup mean. A cell is one arm on one model family, and the "
                      "LLM spread is the mean over cells. The bound is the 95th percentile of the spread when respondents "
                      "are relabeled at random, and the relabeled spread is its mean. The last column states whether the "
                      "profile-mean baseline lies beyond its own bound.")
    s = pd.read_csv(A / "r2_subgroup_structure.csv")
    s = s[s.source == "repeated_splits"]
    body = []
    for ds, name in DS.items():
        g = s[s.dataset == ds]
        cells = []
        for arm in KEY_ARMS:
            v = g[g.arm == arm].correlation
            cells.append(f"{f(v.mean(), 2)} ({f(v.min(), 2)} to {f(v.max(), 2)})")
        cells.append(f(g[g.arm == "profile-center baseline"].correlation.iloc[0], 2))
        body.append(" & ".join([name] + cells) + " \\\\")
    struct = table("Correlation between human and synthetic subgroup differences in scale means, repeated splits. Each "
                   "cell gives the mean over the five model families with the range in parentheses.", "tab:subgroup-structure",
                   "lccccc", "Dataset & DOP & BPP & SADP & HC & Profile mean", body, size="small")
    return [disp, struct]


def t_response_style() -> str:
    r = pd.read_csv(A / "r2_response_style.csv")
    r = r[r.source == "repeated_splits"]
    body = []
    for ds, name in DS.items():
        g = r[r.dataset == ds]
        body.append(panel(name, 6))
        h = g[g.who == "human"].iloc[0]
        dash = lambda s: s.replace("n/d", "--")  # noqa: E731
        body.append(dash(" & ".join(["Human", f(h.share_lowest, 2), f(h.share_highest, 2), f(h.share_midpoint, 2), f(h.entropy, 2),
                                     f(h.straightlining, 2)])) + " \\\\")
        m = g[g.who == "synthetic"].groupby("arm")[["share_lowest", "share_highest", "share_midpoint", "entropy", "straightlining"]].mean()
        for arm in KEY_ARMS:
            body.append(dash(" & ".join([arm] + [f(m.loc[arm, k], 2) for k in m.columns])) + " \\\\")
        body.append("\\midrule")
    return table("Response style of the synthetic and human records, repeated splits, mean over the five model families.",
                 "tab:response-style", "lccccc",
                 "Records & Lowest category & Highest category & Midpoint & Entropy & Straight-lining", body[:-1], size="small",
                 note="Entries are shares of answers, except entropy, which is the entropy of the answer distribution "
                      "divided by its maximum, and straight-lining, which is the share of respondent-by-scale blocks with "
                      "one answer throughout. The Bangladesh response range has no midpoint.")


# --------------------------------------------------------------------------- #
def appendix() -> str:
    grid_tables = t_grid()
    splits = t_splits()
    unc = t_uncertainty()
    base = t_baselines()
    psy = t_psychometric_appendix()
    prof = t_profiles()
    sub = t_subgroup()
    sections = [
        ("Results by model family", grid_tables),
        ("Repeated splits", splits),
        ("Distributional fidelity and uncertainty", [t_distributional(), *unc, t_response_style()]),
        ("Non-LLM baselines", base),
        ("Psychometric properties and measurement invariance", psy),
        ("Profile construction", prof),
        ("Subgroup fidelity", sub),
    ]
    parts = ["\\clearpage", "\\appendix", "\\counterwithin{table}{section}"]
    for i, (title, tables) in enumerate(sections):
        parts.append(f"\\section{{{title}}}\n\\label{{app:{chr(97 + i)}}}")
        parts += tables
        parts.append("\\clearpage")
    return "\n\n".join(parts)


BLOCKS = {"respondent-level": t_respondent_level, "psychometric": t_psychometric_main, "appendix": appendix}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # the public copy of the repository has no manuscript, and the tables are then only written out
    tex = TEX.read_text(encoding="utf-8") if TEX.exists() else ""
    for name, fn in BLOCKS.items():
        block = fn()
        (OUT / f"{name}.tex").write_text(block + "\n", encoding="utf-8")
        begin, end = f"%% BEGIN GENERATED {name}", f"%% END GENERATED {name}"
        if begin in tex and end in tex:
            pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
            tex = pattern.sub(lambda _: f"{begin}\n{block}\n{end}", tex)
            print(f"{name}: block in ipm/manuscript.tex rewritten ({block.count(chr(92) + 'begin{table}')} tables)")
        else:
            print(f"{name}: no markers in the manuscript, written to results/analysis/tables/{name}.tex only")
    if TEX.exists():
        TEX.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
