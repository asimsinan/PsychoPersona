"""Figures added in the round-2 revision.

recovery      Profile recovery of DOP, BPP and SADP on the final-test split, five model
              families, oracle and cold-start assignment, against the majority-profile
              rate in the same label space.
item_mae      Item MAE on the final-test split with 95% bootstrap intervals over
              respondents: the eight LLM arms and the non-LLM baselines, with the
              human same-profile reference.
tradeoff      Distribution-level distance (TVD) against respondent-level error (item
              MAE) for the LLM arms and the non-LLM baselines.
splits        Item MAE and profile recovery of every arm and model family across the
              ten repeated splits, against the profile-mean baseline and the majority
              rate on the same splits.

Every figure is drawn from a table that is written next to it
(``results/analysis/figures/<name>.csv``), so each plotted value can be checked. Output
goes to ``ipm/`` as PNG at 500 dpi and as PDF.

The five model families use one colour and one marker each throughout. The colours pass
the colour-vision checks of the palette validator for every pair; the marker shape is the
second cue.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_figures.py [name ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import r2_common as C  # noqa: E402
from compute_nonllm_baselines import _columns, _dataset_settings, _evaluate, _recompute_scale_means  # noqa: E402
from r2_profile_sensitivity import recovery_labels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
DATA_OUT = ANALYSIS / "figures"
# next to the manuscript when there is one, otherwise with the plotted values
FIG_OUT = REPO / "ipm" if (REPO / "ipm").is_dir() else DATA_OUT
WIDTH = 6.5                       # text width of the manuscript, inches
DATASETS = {"anket_tr": "Turkish dataset", "bangladesh_sns": "Bangladesh dataset"}
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#6b6a66"
FAMILY = {                        # colour, marker
    "GPT-5.4": ("#2a78d6", "o"),
    "Claude Opus 4.6": ("#e34948", "s"),
    "Gemini 3.1 Pro": ("#eda100", "^"),
    "Qwen3.8 27B": ("#008300", "D"),
    "DeepSeek V4.1 Flash": ("#1baf7a", "v"),
}
HOSTED = ["GPT-5.4", "Claude Opus 4.6", "Gemini 3.1 Pro"]
ARMS = C.ARM_ORDER
BOOT = 2000
SEED = 20261112

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 500, "figure.dpi": 150,
})


def save(fig, name: str, table: pd.DataFrame) -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATA_OUT / f"{name}.csv", index=False)
    fig.savefig(FIG_OUT / f"fig_{name}.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG_OUT / f"fig_{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"fig_{name}: {len(table)} plotted values -> {FIG_OUT.relative_to(REPO)}/fig_{name}.png, results/analysis/figures/{name}.csv")


def family_handles(families=FAMILY, size=4.5) -> list[Line2D]:
    return [Line2D([], [], linestyle="none", marker=FAMILY[f][1], markersize=size, markerfacecolor=FAMILY[f][0],
                   markeredgecolor="white", markeredgewidth=0.4, label=f) for f in families]


def final_test_frames(run: str, ds: str):
    """Human and synthetic final-test records of one run, both with the assigned profile."""
    cols = _columns(_dataset_settings(ds))
    human, synthetic, _ = C.load_run(run, ds)
    human = _recompute_scale_means(human, cols)
    synthetic = _recompute_scale_means(synthetic, cols)
    synthetic["profile_id"] = human.profile_id.values
    return human, synthetic, cols


# --------------------------------------------------------------------------- #
def fig_recovery() -> None:
    rows = []
    for assignment, path in (("oracle", "run_index.csv"), ("cold-start", "coldstart_run_index.csv")):
        idx = pd.read_csv(ANALYSIS / path)
        idx = idx[(idx.split == "final_test") & idx.arm.isin(["DOP", "BPP", "SADP"])]
        for r in idx.itertuples():
            human, synthetic, cols = final_test_frames(r.run, r.dataset)
            value = _evaluate(human, synthetic, cols)["PRA"]
            assigned = int(human.profile_id.nunique())
            majority = (float(pd.Series(recovery_labels(human, human, cols.scale_columns)).value_counts(normalize=True).max())
                        if assigned > 1 else np.nan)
            rows.append({"dataset": r.dataset, "arm": r.arm, "assignment": assignment, "model": r.model,
                         "recovery": value, "majority": majority, "profiles_assigned": assigned,
                         "n": len(human), "run": r.run})
    table = pd.DataFrame(rows)
    layout = [("DOP", "oracle", "DOP"), ("BPP", "oracle", "BPP, oracle"), ("SADP", "oracle", "SADP, oracle"),
              ("BPP", "cold-start", "BPP, cold-start"), ("SADP", "cold-start", "SADP, cold-start")]
    offsets = np.linspace(0.26, -0.26, len(FAMILY))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), sharey=True, sharex=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        for y, (arm, assignment, _) in enumerate(layout):
            cell = table[(table.dataset == ds) & (table.arm == arm) & (table.assignment == assignment)]
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f4f3f0", linewidth=0, zorder=0)
            if cell.recovery.isna().all():
                ax.text(0.62, y, "not defined, one profile assigned", va="center", ha="center",
                        fontsize=6.5, style="italic", color=INK2)
                continue
            ax.plot([cell.majority.iloc[0]] * 2, [y - 0.42, y + 0.42], color=INK, linewidth=1.1, zorder=2)
            for off, (family, (colour, marker)) in zip(offsets, FAMILY.items()):
                v = cell[cell.model == family]
                if len(v):
                    ax.plot(v.recovery.iloc[0], y + off, linestyle="none", marker=marker, markersize=4.6,
                            markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlim(0.25, 1.0)
        ax.set_ylim(len(layout) - 0.5, -0.5)
        ax.set_yticks(range(len(layout)), [lab for *_, lab in layout])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Profile recovery")
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    handles = family_handles() + [Line2D([], [], color=INK, linewidth=1.1, label="Majority-profile rate")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.115), columnspacing=1.1,
               handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "recovery", table)


# --------------------------------------------------------------------------- #
BASELINE_ROWS = [  # method in r2_baselines_paper.csv, label, group
    ("profile_center", "Profile mean", "Non-LLM, profile known"),
    ("ordreg_profile_metadata_median", "Ordinal regression", "Non-LLM, profile known"),
    ("within_profile", "Within-profile sampler", "Non-LLM, profile known"),
    ("grm_profile", "Graded response model", "Non-LLM, profile known"),
    ("copula_profile", "Gaussian copula", "Non-LLM, profile known"),
    ("ordreg_metadata_median", "Ordinal regression", "Non-LLM, metadata only"),
    ("nn_metadata", "Nearest neighbor", "Non-LLM, metadata only"),
    ("marginal", "Marginal sampler", "Non-LLM, no respondent information"),
]


def llm_respondent_mae() -> pd.DataFrame:
    """Per-respondent item MAE of every final-test oracle run of the paper's split."""
    idx = pd.read_csv(ANALYSIS / "run_index.csv")
    out = []
    for r in idx[idx.split == "final_test"].itertuples():
        human, synthetic, cols = final_test_frames(r.run, r.dataset)
        mae = np.abs(synthetic[cols.item_columns].to_numpy(dtype=float)
                     - human[cols.item_columns].to_numpy(dtype=float)).mean(axis=1)
        out.append(pd.DataFrame({"dataset": r.dataset, "method": r.arm, "model": r.model,
                                 "respondent_id": human.respondent_id.values, "item_mae": mae}))
    return pd.concat(out, ignore_index=True)


def fig_item_mae() -> None:
    rng = np.random.default_rng(SEED)
    llm = llm_respondent_mae()
    base = pd.read_csv(ANALYSIS / "r2_baselines_paper_respondents.csv")
    base = base[base.setting == "oracle"].assign(model="non-LLM")
    rows = []
    for (ds, method, model), g in pd.concat([llm, base], ignore_index=True).groupby(["dataset", "method", "model"]):
        lo, hi = C.bootstrap_ci(g.item_mae.to_numpy(), rng, BOOT)
        rows.append({"dataset": ds, "method": method, "model": model, "n": len(g),
                     "item_mae": g.item_mae.mean(), "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows)
    # The intervals the manuscript tabulates (uncertainty_summary.csv) are used where they exist, so that the
    # figure and the table agree to the last digit. They come from the same procedure with another random stream.
    stored = pd.read_csv(ANALYSIS / "uncertainty_summary.csv")
    stored = stored[stored.metric == "ItemMAE"].replace({"arm": {"Profile-center": "profile_center"}})
    for s in stored.itertuples():
        hit = (table.dataset == s.dataset) & (table.method == s.arm) & table.model.isin(["GPT-5.4", "non-LLM"])
        if abs(table.loc[hit, "item_mae"].iloc[0] - s.mean) > 1e-6:
            raise RuntimeError(f"stored mean differs for {s.dataset} {s.arm}")
        table.loc[hit, ["ci_lo", "ci_hi"]] = [s.ci_lo, s.ci_hi]
    ceiling = pd.read_csv(ANALYSIS / "r2_noise_ceiling.csv")
    ceiling = ceiling[(ceiling.source == "paper_split") & (ceiling.split == "final_test")].set_index("dataset")
    table["human_same_profile"] = table.dataset.map(ceiling.human_vs_same_profile)

    # row layout shared by the two panels: (key, label) or a group header
    layout: list[tuple] = [("header", "LLM arms")] + [("llm", a, a) for a in ARMS]
    group = None
    for method, label, grp in BASELINE_ROWS:
        if grp != group:
            layout.append(("header", grp))
            group = grp
        layout.append(("base", method, label))
    ypos = np.arange(len(layout))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.3), sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        t = table[table.dataset == ds]
        ref = float(ceiling.loc[ds, "human_vs_same_profile"])
        center = float(t[(t.method == "profile_center")].item_mae.iloc[0])
        ax.axvline(ref, color=INK2, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
        ax.axvline(center, color=INK2, linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=1)
        for y, entry in zip(ypos, layout):
            if entry[0] == "header":
                continue
            if entry[0] == "llm":
                cell = t[t.method == entry[1]]
                for family, (colour, marker) in FAMILY.items():
                    v = cell[cell.model == family]
                    if not len(v):
                        continue
                    v = v.iloc[0]
                    if family == "GPT-5.4":
                        ax.plot([v.ci_lo, v.ci_hi], [y, y], color=colour, linewidth=1.0, zorder=3, solid_capstyle="butt")
                        ax.plot(v.item_mae, y, linestyle="none", marker=marker, markersize=4.6, markerfacecolor=colour,
                                markeredgecolor="white", markeredgewidth=0.4, zorder=5)
                    else:
                        ax.plot(v.item_mae, y, linestyle="none", marker=marker, markersize=3.6, markerfacecolor="none",
                                markeredgecolor=colour, markeredgewidth=0.8, zorder=4)
            else:
                v = t[t.method == entry[1]].iloc[0]
                ax.plot([v.ci_lo, v.ci_hi], [y, y], color=NEUTRAL, linewidth=1.0, zorder=3, solid_capstyle="butt")
                ax.plot(v.item_mae, y, linestyle="none", marker="s", markersize=4.2, markerfacecolor=NEUTRAL,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax.set_title(title, loc="left", pad=14)
        top = ax.get_xaxis_transform()
        ax.text(ref, 1.005, "human,\nsame profile", transform=top, ha="center", va="bottom", fontsize=6, color=INK2,
                linespacing=0.95)
        ax.text(center, 1.005, "profile\nmean", transform=top, ha="center", va="bottom", fontsize=6, color=INK2,
                linespacing=0.95)
        ax.set_ylim(len(layout) - 0.5, -0.6)
        ax.set_xlabel("Item MAE (lower is better)")
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos, [e[1] if e[0] == "header" else e[2] for e in layout])
    for tick, entry in zip(axes[0].get_yticklabels(), layout):
        if entry[0] == "header":
            tick.set_fontweight("bold")
            tick.set_color(INK)
            tick.set_fontsize(7)
    handles = [Line2D([], [], color=FAMILY["GPT-5.4"][0], linewidth=1.0, marker="o", markersize=4.6,
                      markerfacecolor=FAMILY["GPT-5.4"][0], markeredgecolor="white", markeredgewidth=0.4,
                      label="GPT-5.4, 95% interval")]
    handles += [Line2D([], [], linestyle="none", marker=FAMILY[f][1], markersize=3.6, markerfacecolor="none",
                       markeredgecolor=FAMILY[f][0], markeredgewidth=0.8, label=f) for f in list(FAMILY)[1:]]
    handles += [Line2D([], [], color=NEUTRAL, linewidth=1.0, marker="s", markersize=4.2, markerfacecolor=NEUTRAL,
                       markeredgecolor="white", markeredgewidth=0.4, label="Non-LLM baseline, 95% interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.075), columnspacing=1.2,
               handletextpad=0.4)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "item_mae", table)


# --------------------------------------------------------------------------- #
def fig_tradeoff() -> None:
    dist = pd.read_csv(ANALYSIS / "simbench_distributional.csv")
    dist = dist[(dist.split == "final_test") & dist.model.isin(HOSTED)]
    llm = llm_respondent_mae()
    llm = llm[llm.model.isin(HOSTED)].groupby(["dataset", "method", "model"]).item_mae.mean().reset_index()
    arms = llm.merge(dist.rename(columns={"arm": "method"})[["dataset", "method", "model", "TVD"]],
                     on=["dataset", "method", "model"])
    arms = arms.groupby(["dataset", "method"]).agg(item_mae=("item_mae", "mean"), tvd=("TVD", "mean"),
                                                   families=("model", "nunique")).reset_index()
    arms["kind"] = "LLM arm"
    base = pd.read_csv(ANALYSIS / "r2_baselines_paper.csv")
    base = base[base.setting == "oracle"]
    keep = {"profile_center": "Profile mean", "ordreg_profile_metadata_median": "Ordinal regression,\nprofile and metadata",
            "ordreg_metadata_median": "Ordinal regression,\nmetadata", "within_profile": "Within-profile sampler",
            "grm_profile": "Graded response model", "copula_profile": "Gaussian copula", "marginal": "Marginal sampler"}
    base = base[base.method.isin(keep)].assign(item_mae=lambda d: d.RespItemMAE, tvd=lambda d: d.TVD,
                                               families=0, kind="non-LLM baseline")
    table = pd.concat([arms, base[["dataset", "method", "item_mae", "tvd", "families", "kind"]]], ignore_index=True)

    blue = FAMILY["GPT-5.4"][0]
    samplers = ["within_profile", "grm_profile", "copula_profile"]
    # label offsets in points (dx, dy, horizontal anchor, vertical anchor), set so that no two labels collide
    nudge = {"DOP": (6, 0, "left", "center"), "HC": (-6, 0, "right", "center"),
             "profile_center": (0, 6, "center", "bottom"),
             "ordreg_profile_metadata_median": (6, 0, "left", "center"),
             "ordreg_metadata_median": (6, 0, "left", "center"), "marginal": (0, 6, "center", "bottom")}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        t = table[table.dataset == ds]
        a, b = t[t.kind == "LLM arm"], t[t.kind != "LLM arm"]
        cluster = a[~a.method.isin(["DOP", "HC"])]
        ax.plot(b.item_mae, b.tvd, linestyle="none", marker="s", markersize=4.4, markerfacecolor=NEUTRAL,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.plot(a.item_mae, a.tvd, linestyle="none", marker="o", markersize=4.8, markerfacecolor=blue,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        for r in t.itertuples():
            if r.method not in nudge:
                continue
            dx, dy, ha, va = nudge[r.method]
            ax.annotate(keep.get(r.method, r.method), (r.item_mae, r.tvd), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=6.3, color=INK2 if r.kind != "LLM arm" else INK, linespacing=0.95)
        # the six persona and statistics-aware arms and the three samplers are labelled as groups
        ax.annotate("BPP, FPNP, BBP,\nSCP, SADP, TPSAP", (cluster.item_mae.max(), cluster.tvd.mean()), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.3, color=INK, linespacing=0.95)
        s = b[b.method.isin(samplers)]
        ax.annotate("Within-profile sampler,\ngraded response model,\nGaussian copula", (s.item_mae.mean(), s.tvd.min()),
                    xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=6.3, color=INK2,
                    linespacing=0.95)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel("Item MAE, respondent level (lower is better)")
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.02, 0.56)
        ax.margins(x=0.17)
    axes[0].set_ylabel("Total variation distance,\ndistribution level (lower is better)")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4.8, markerfacecolor=blue, markeredgecolor="white",
                      markeredgewidth=0.4, label="LLM arm, mean of the three hosted families"),
               Line2D([], [], linestyle="none", marker="s", markersize=4.4, markerfacecolor=NEUTRAL,
                      markeredgecolor="white", markeredgewidth=0.4, label="Non-LLM baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), columnspacing=1.6,
               handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "tradeoff", table)


# --------------------------------------------------------------------------- #
def fig_splits() -> None:
    runs = pd.read_csv(ANALYSIS / "c2_llm_runs.csv")
    table = runs.rename(columns={"respondent_item_mae_mean": "item_mae", "profile_recovery_accuracy": "recovery"})[
        ["dataset", "model", "arm", "seed", "item_mae", "recovery", "run"]]
    center = pd.read_csv(ANALYSIS / "r2_baselines_llm_splits.csv")
    center = center[center.method == "profile_center"].set_index(["dataset", "seed"]).RespItemMAE
    resc = pd.read_csv(ANALYSIS / "r2_profile_rescoring.csv")
    resc = resc[(resc.method == "pipeline") & (resc.source == "repeated_splits")]
    majority = resc.groupby(["dataset", "seed"]).majority.first()
    table["profile_mean_item_mae"] = [center[(r.dataset, r.seed)] for r in table.itertuples()]
    table["majority"] = [majority[(r.dataset, r.seed)] for r in table.itertuples()]

    offsets = np.linspace(-0.3, 0.3, len(FAMILY))
    metrics = [("item_mae", "Item MAE", "profile_mean_item_mae", "profile mean"),
               ("recovery", "Profile recovery", "majority", "majority-profile rate")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5), sharex=True, sharey="row")
    for i, (metric, label, ref, ref_label) in enumerate(metrics):
        for j, (ds, title) in enumerate(DATASETS.items()):
            ax = axes[i, j]
            t = table[table.dataset == ds]
            per_split = t.groupby("seed")[ref].first()
            ax.axhspan(per_split.min(), per_split.max(), color="#ecebe7", linewidth=0, zorder=0)
            ax.axhline(per_split.mean(), color=INK2, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
            for x, arm in enumerate(ARMS):
                single = t[t.arm == arm].model.nunique() == 1          # arms run on GPT-5.4 only sit in the middle
                for off, (family, (colour, marker)) in zip(offsets, FAMILY.items()):
                    v = t[(t.arm == arm) & (t.model == family)][metric]
                    if not len(v):
                        continue
                    off = 0.0 if single else off
                    ax.plot(np.full(len(v), x + off), v, linestyle="none", marker=marker, markersize=2.6,
                            markerfacecolor=colour, markeredgecolor="none", alpha=0.55, zorder=3)
                    ax.plot([x + off - 0.055, x + off + 0.055], [v.mean()] * 2, color=INK, linewidth=1.1, zorder=4)
            if i == 0:
                ax.set_title(title, loc="left", pad=4)
            if j == 0:
                ax.set_ylabel(label + (" (lower is better)" if metric == "item_mae" else ""))
            if j == 1:
                ax.text(1.01, per_split.mean(), ref_label.replace(" ", "\n", 1) if metric == "recovery" else ref_label.replace(" ", "\n"),
                        transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6, color=INK2, linespacing=0.95)
            ax.set_xticks(range(len(ARMS)), ARMS)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.55, len(ARMS) - 0.45)
    handles = family_handles(size=4.0) + [Line2D([], [], color=INK, linewidth=1.1, label="Mean of ten splits")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.1,
               handletextpad=0.3)
    fig.subplots_adjust(wspace=0.06, hspace=0.12)
    save(fig, "splits", table)


FIGURES = {"recovery": fig_recovery, "item_mae": fig_item_mae, "tradeoff": fig_tradeoff, "splits": fig_splits}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
