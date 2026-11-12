"""Stronger non-LLM baselines under the benchmark protocol (Reviewer 1, comment 2).

Adds three conventional model classes for ordinal questionnaire data to the four
baselines the manuscript already reports, and scores all of them with the evaluation
code the LLM arms go through. Every model is fitted on the training split alone.

Each new baseline is matched to an information set, so that it can be read against the
LLM arm that receives the same information about a held-out respondent.

nothing about the respondent   grm_population, copula_population (and marginal)
profile label                  grm_profile, copula_profile (and profile_center,
                               within_profile)
pre-response metadata          ordreg_metadata, ordreg_metadata_median (and nn_metadata)
profile label and metadata     ordreg_profile_metadata, ordreg_profile_metadata_median

The profile label of a held-out respondent is the oracle assignment or the cold-start
prediction from metadata, exactly as for the LLM arms. Models are fitted once per
training split and generate under both settings.

Two analyses are written.

paper   the split the manuscript's tables report (seed 42, final test). Stochastic
        baselines are averaged over 50 replicates.
splits  the 50 random splits of ``split_robustness.py`` (seeds 0-49), one draw per
        split, with profiles refitted on every training split.
llm     the ten splits on which the LLM arms were re-run (seeds 101-110), so that every
        baseline can be compared with every arm and model family split by split. The
        split and the profile assignment are checked against the stored LLM runs.

The four existing baselines are recomputed alongside and compared with the files the
manuscript's numbers come from, so a difference in data, split or profile fit would show.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_stronger_baselines.py [--splits 50] [--workers 10]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r2_ordinal_models as M  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from compute_nonllm_baselines import (  # noqa: E402
    METRIC_KEYS, _columns, _dataset_settings, _evaluate, _impute_train_items, _load_run,
    _recompute_scale_means, baseline_marginal, baseline_nn_metadata, baseline_profile_center,
    baseline_within_profile,
)
from saykopersona.coldstart import fit_coldstart_predictor, predict_coldstart_profiles  # noqa: E402
from saykopersona.data import build_stratified_three_way_split, load_raw_survey, score_survey_dataframe  # noqa: E402
from saykopersona.profiles import fit_empirical_profiles, predict_profiles  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
RUNS = REPO / "results" / "runs" / "experiment"
DATA = {"anket_tr": REPO / "datasets" / "anket.xlsx",
        "bangladesh_sns": REPO / "datasets" / "bangladesh_sns" / "mmc2.xlsx"}
FEATURES = {"anket_tr": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"],
            "bangladesh_sns": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean"]}
DROP_METADATA = {"timestamp", "platforms", "primary_platform"}     # as in the nn_metadata baseline
GRID = np.linspace(-4.0, 4.0, 41)
PAPER_REPLICATES = 50
SEED = 20261112

OLD_DETERMINISTIC = {"profile_center": baseline_profile_center, "nn_metadata": baseline_nn_metadata}
OLD_STOCHASTIC = {"within_profile": baseline_within_profile, "marginal": baseline_marginal}
NEW_STOCHASTIC = ["grm_population", "grm_profile", "copula_population", "copula_profile",
                  "ordreg_metadata", "ordreg_profile_metadata"]
NEW_DETERMINISTIC = ["ordreg_metadata_median", "ordreg_profile_metadata_median"]
INFORMATION = {"marginal": "none", "grm_population": "none", "copula_population": "none",
               "profile_center": "profile", "within_profile": "profile", "grm_profile": "profile",
               "copula_profile": "profile", "nn_metadata": "metadata", "ordreg_metadata": "metadata",
               "ordreg_metadata_median": "metadata", "ordreg_profile_metadata": "profile and metadata",
               "ordreg_profile_metadata_median": "profile and metadata"}
ORDER = ["marginal", "grm_population", "copula_population", "profile_center", "within_profile",
         "grm_profile", "copula_profile", "nn_metadata", "ordreg_metadata", "ordreg_metadata_median",
         "ordreg_profile_metadata", "ordreg_profile_metadata_median"]


# --------------------------------------------------------------------------- #
# model fitting and generation                                                 #
# --------------------------------------------------------------------------- #
class Design:
    """One-hot covariates, with levels and scaling fixed on the training split."""

    def __init__(self, train: pd.DataFrame, columns: list[str]):
        self.columns = columns
        dummies = pd.get_dummies(train[columns].astype(str), dummy_na=False).astype(float)
        keep = dummies.columns[dummies.std(axis=0) > 0]
        self.levels = list(keep)
        self.mean = dummies[keep].mean(axis=0)
        self.std = dummies[keep].std(axis=0)

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        dummies = pd.get_dummies(frame[self.columns].astype(str), dummy_na=False).astype(float)
        dummies = dummies.reindex(columns=self.levels, fill_value=0.0)
        return ((dummies - self.mean) / self.std).to_numpy()


class Models:
    """All new baselines, fitted on one training split."""

    def __init__(self, train: pd.DataFrame, cols, seed: int):
        rng = np.random.default_rng([SEED, seed, 0])
        self.cols = cols
        self.items = cols.item_columns
        raw = train[self.items].astype(float).to_numpy()
        self.categories = [np.unique(raw[:, j]) for j in range(raw.shape[1])]
        self.k = [len(c) for c in self.categories]
        x = M.encode_items(raw, self.categories)
        profiles = train["profile_id"].astype(int).to_numpy()
        self.diag: dict = {}

        # graded response model per scale, then the latent distribution per group
        position = {item: j for j, item in enumerate(self.items)}
        self.scale_index = [[position[i] for i in items] for items in cols.scale_to_items.values()]
        self.grm = []
        logliks = []
        for s, idx in enumerate(self.scale_index):
            model = M.GradedResponse().fit(x[:, idx], [self.k[j] for j in idx])
            self.grm.append(model)
            logliks.append(model.loglik_on(x[:, idx], GRID))
            self.diag[f"grm_converged_{s}"] = model.converged_
            self.diag[f"grm_cycles_{s}"] = model.cycles_
            self.diag[f"grm_min_slope_{s}"] = float(model.a_.min())
        pi, cycles, _ = M.latent_histogram(logliks, GRID)
        self.latent = {"population": pi}
        self.diag["latent_cycles_population"] = cycles
        for label in np.unique(profiles):
            sel = profiles == label
            pi, cycles, _ = M.latent_histogram([ll[sel] for ll in logliks], GRID)
            self.latent[int(label)] = pi
            self.diag[f"latent_cycles_{int(label)}"] = cycles

        # Gaussian copula: one group, and thresholds per profile with a common correlation
        self.copula_population = M.OrdinalCopula().fit(x, np.zeros(len(x), dtype=int), self.k)
        self.copula_profile = M.OrdinalCopula().fit(x, profiles, self.k)
        for name, c in (("population", self.copula_population), ("profile", self.copula_profile)):
            self.diag[f"copula_{name}_min_eigenvalue"] = c.min_eigenvalue_
            self.diag[f"copula_{name}_share_at_bound"] = c.at_bound_

        # proportional-odds regression on metadata, without and with the profile label
        meta = [c for c in cols.metadata_columns if c in train.columns and c not in DROP_METADATA]
        self.design_meta = Design(train, meta)
        self.design_both = Design(train, [*meta, "profile_id"])
        self.ordreg_meta = M.OrdinalRegression().fit(self.design_meta(train), x, rng)
        self.ordreg_both = M.OrdinalRegression().fit(self.design_both(train), x, rng)
        self.diag["ordreg_metadata_penalty"] = self.ordreg_meta.lam_
        self.diag["ordreg_profile_metadata_penalty"] = self.ordreg_both.lam_
        self.diag["ordreg_columns"] = len(self.design_meta.levels)

    # ------------------------------------------------------------------ #
    def _frame(self, human: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
        out = human[["respondent_id", "profile_id"]].reset_index(drop=True).copy()
        values = np.column_stack([self.categories[j][index[:, j]] for j in range(index.shape[1])])
        out[self.items] = values
        return _recompute_scale_means(out, self.cols)

    def _grm(self, groups: list, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((len(groups), len(self.items)), dtype=int)
        theta = np.empty((len(groups), len(self.grm)))
        groups = np.asarray(groups, dtype=object)
        for label in set(groups.tolist()):
            sel = np.array([g == label for g in groups])
            theta[sel] = M.sample_latent(self.latent[label], GRID, int(sel.sum()), rng)
        for s, idx in enumerate(self.scale_index):
            out[:, idx] = self.grm[s].sample(theta[:, s], rng)
        return out

    def stochastic(self, name: str, human: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        profile = human["profile_id"].astype(int).to_numpy()
        if name == "grm_population":
            index = self._grm(["population"] * len(human), rng)
        elif name == "grm_profile":
            index = self._grm(profile.tolist(), rng)
        elif name == "copula_population":
            index = self.copula_population.sample(np.zeros(len(human), dtype=int), rng)
        elif name == "copula_profile":
            index = self.copula_profile.sample(profile, rng)
        elif name == "ordreg_metadata":
            index = self.ordreg_meta.sample(self.design_meta(human), rng)
        elif name == "ordreg_profile_metadata":
            index = self.ordreg_both.sample(self.design_both(human), rng)
        else:
            raise ValueError(name)
        return self._frame(human, index)

    def deterministic(self, name: str, human: pd.DataFrame) -> pd.DataFrame:
        if name == "ordreg_metadata_median":
            return self._frame(human, self.ordreg_meta.median(self.design_meta(human)))
        if name == "ordreg_profile_metadata_median":
            return self._frame(human, self.ordreg_both.median(self.design_both(human)))
        raise ValueError(name)


def _mean_sd(rows: list[dict]) -> dict:
    out = {}
    for key in METRIC_KEYS:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[key] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{key}_sd"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- #
# analysis 1: the paper's split                                                #
# --------------------------------------------------------------------------- #
def _tvd(human: pd.DataFrame, sim: pd.DataFrame, cols, lo: int, hi: int) -> float:
    """Mean total variation distance between human and synthetic answer distributions, per item.

    Same definition as ``simbench_distributional.py``.
    """
    out = []
    for item in cols.item_columns:
        p = np.bincount(human[item].dropna().astype(int) - lo, minlength=hi - lo + 1) / human[item].notna().sum()
        q = np.bincount(sim[item].astype(int) - lo, minlength=hi - lo + 1) / len(sim)
        out.append(0.5 * np.abs(p - q).sum())
    return float(np.mean(out))


def paper_split(dataset: str) -> tuple[list[dict], dict, list[dict]]:
    warnings.filterwarnings("ignore")
    settings = _dataset_settings(dataset)
    cols = _columns(settings)
    lo, hi = settings.response_min, settings.response_max
    idx = pd.read_csv(ANALYSIS / "run_index.csv")
    run = idx[(idx.dataset == dataset) & (idx.model == "GPT-5.4") & (idx.arm == "SADP")
              & (idx.split == "final_test")].run.iloc[0]
    human, train = _load_run(RUNS / run, cols)
    human = _recompute_scale_means(human, cols)
    train = _recompute_scale_means(_impute_train_items(train, cols), cols)
    cold = pd.read_csv(ANALYSIS / "coldstart_assignments" / f"{dataset}_final_test_coldstart.csv")
    cold = cold.set_index("respondent_id").loc[human.respondent_id]
    if not (cold.profile_id_oracle.values == human.profile_id.values).all():
        raise RuntimeError(f"{dataset}: stored cold-start file does not match the run's oracle assignment")
    models = Models(train, cols, seed=42)
    truth = human[cols.item_columns].to_numpy(dtype=float)

    rows, respondents = [], []
    for setting in ("oracle", "coldstart"):
        frame = human.copy()
        if setting == "coldstart":
            frame["profile_id"] = cold.profile_id_coldstart.values
        base = {"dataset": dataset, "setting": setting, "run": run, "n": len(frame)}

        def add(name: str, sims: list[pd.DataFrame]) -> None:
            reps = [{**_evaluate(frame, sim, cols), "TVD": _tvd(frame, sim, cols, lo, hi)} for sim in sims]
            if len(reps) == 1:
                rows.append({**base, "method": name, "replicates": 1, **reps[0]})
            else:
                tvd = [r["TVD"] for r in reps]
                rows.append({**base, "method": name, "replicates": len(reps), **_mean_sd(reps),
                             "TVD": float(np.mean(tvd)), "TVD_sd": float(np.std(tvd))})
            # error of each respondent, averaged over the replicates
            mae = np.mean([np.abs(sim[cols.item_columns].to_numpy(dtype=float) - truth).mean(axis=1)
                           for sim in sims], axis=0)
            respondents.extend({"dataset": dataset, "setting": setting, "method": name,
                                "respondent_id": rid, "item_mae": float(v)}
                               for rid, v in zip(frame.respondent_id, mae))

        for name, fn in OLD_DETERMINISTIC.items():
            add(name, [fn(frame, train, cols)])
        for name, fn in OLD_STOCHASTIC.items():      # replicate seeds as in compute_nonllm_baselines.py
            add(name, [fn(frame, train, cols, np.random.default_rng(s)) for s in range(PAPER_REPLICATES)])
        for m, name in enumerate(NEW_STOCHASTIC):
            add(name, [models.stochastic(name, frame, np.random.default_rng([SEED, 42, m + 1, s]))
                       for s in range(PAPER_REPLICATES)])
        for name in NEW_DETERMINISTIC:
            add(name, [models.deterministic(name, frame)])
    return rows, {"dataset": dataset, "analysis": "paper", "seed": 42, **models.diag}, respondents


# --------------------------------------------------------------------------- #
# analysis 2: fifty random splits                                              #
# --------------------------------------------------------------------------- #
def one_split(task: tuple) -> tuple[list[dict], dict]:
    warnings.filterwarnings("ignore")
    dataset, seed, check_run = task
    settings = _dataset_settings(dataset)
    cols = _columns(settings)
    scored = score_survey_dataframe(load_raw_survey(DATA[dataset], dataset=settings), dataset=settings)
    train, _val, test = build_stratified_three_way_split(
        scored, validation_size=0.1, final_test_size=0.1, random_state=seed, dataset=settings)
    result = fit_empirical_profiles(train.dropna(subset=FEATURES[dataset]), feature_columns=FEATURES[dataset],
                                    random_state=seed, anchor_columns=list(settings.profile_anchor_columns))
    test_p = test.copy()
    test_p["profile_id"] = predict_profiles(result, test_p.dropna(subset=FEATURES[dataset])).reindex(test_p.index)
    test_p = test_p.dropna(subset=["profile_id"])
    test_p["profile_id"] = test_p["profile_id"].astype(int)
    predictor = fit_coldstart_predictor(result.assignments, dataset=settings, random_state=seed)
    cold = predict_coldstart_profiles(predictor, test_p.copy())
    train_p = _recompute_scale_means(_impute_train_items(result.assignments.copy(), cols), cols)
    test_p = _recompute_scale_means(test_p, cols)
    models = Models(train_p, cols, seed=seed)
    if check_run:                       # same held-out respondents and profiles as the stored LLM run
        stored = pd.read_csv(RUNS / check_run / "profiles" / "final_test_profile_assignments.csv")
        stored = stored.set_index("respondent_id")["profile_id"]
        same = set(stored.index) == set(test_p.respondent_id)
        # the mixture numbers its components arbitrarily, so partitions are compared, not labels
        models.diag["matches_llm_run"] = bool(
            same and adjusted_rand_score(stored.loc[test_p.respondent_id].to_numpy(), test_p.profile_id.to_numpy()) == 1.0)

    rows = []
    for setting in ("oracle", "coldstart"):
        frame = test_p.copy()
        if setting == "coldstart":
            frame["profile_id"] = cold.values
        base = {"dataset": dataset, "seed": seed, "setting": setting, "selected_k": result.selected_k,
                "n_test": len(frame), "coldstart_model": predictor.selected,
                "coldstart_agreement": float((cold.values == test_p.profile_id.values).mean())}
        rng = np.random.default_rng(20260907 + seed)        # same stream and order as split_robustness.py
        sims = [("profile_center", baseline_profile_center(frame, train_p, cols)),
                ("within_profile", baseline_within_profile(frame, train_p, cols, rng)),
                ("nn_metadata", baseline_nn_metadata(frame, train_p, cols)),
                ("marginal", baseline_marginal(frame, train_p, cols, rng))]
        for m, name in enumerate(NEW_STOCHASTIC):
            sims.append((name, models.stochastic(name, frame, np.random.default_rng([SEED, seed, m + 1]))))
        for name in NEW_DETERMINISTIC:
            sims.append((name, models.deterministic(name, frame)))
        rows += [{**base, "method": name, **_evaluate(frame, sim, cols)} for name, sim in sims]
    return rows, {"dataset": dataset, "analysis": "llm" if check_run else "splits", "seed": seed, **models.diag}


# --------------------------------------------------------------------------- #
def _check(label: str, new: pd.DataFrame, old: pd.DataFrame, keys: list[str], metrics: list[str]) -> None:
    merged = new.merge(old, on=keys, suffixes=("", "_old"))
    worst = max(float((merged[m] - merged[f"{m}_old"]).abs().max()) for m in metrics)
    print(f"  {label}: {len(merged)} rows compared, largest absolute difference {worst:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-splits", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    show = ["RespItemMAE", "RespScaleRMSE", "ItemEMR", "CorrRMSE", "WithinRMSE", "PRA", "SimUniqueRatio"]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        paper_jobs = [pool.submit(paper_split, ds) for ds in DATA]
        split_jobs = [] if args.skip_splits else [pool.submit(one_split, (ds, s, None)) for ds in DATA for s in range(args.splits)]
        llm_runs = pd.read_csv(ANALYSIS / "c2_llm_runs.csv")
        first = llm_runs.groupby(["dataset", "seed"]).run.first()
        llm_jobs = [pool.submit(one_split, (ds, int(s), run)) for (ds, s), run in first.items()]
        paper_out = [j.result() for j in paper_jobs]
        split_out = [j.result() for j in split_jobs]
        llm_out = [j.result() for j in llm_jobs]

    paper = pd.DataFrame([r for rows, _, _ in paper_out for r in rows])
    paper.insert(3, "information", paper.method.map(INFORMATION))
    paper.to_csv(ANALYSIS / "r2_baselines_paper.csv", index=False)
    pd.DataFrame([r for _, _, resp in paper_out for r in resp]).round(6).to_csv(
        ANALYSIS / "r2_baselines_paper_respondents.csv", index=False)
    diag = [d for _, d, _ in paper_out]

    print("=== the paper's split, final test ===")
    for (ds, setting), g in paper.groupby(["dataset", "setting"], sort=False):
        print(f"\n{ds}, {setting} assignment (n = {g.n.iloc[0]})")
        print(g.set_index("method").loc[ORDER, ["information", "replicates", *show, "TVD"]].round(3).to_string())

    print("\n=== check of the recomputed existing baselines against the manuscript's source files ===")
    for ds in DATA:
        old = pd.read_csv(ANALYSIS / f"nonllm_baselines_{ds}_sadp.csv")
        _check(f"{ds} oracle, nonllm_baselines_{ds}_sadp.csv", paper[(paper.dataset == ds) & (paper.setting == "oracle")],
               old, ["method"], ["RespItemMAE", "RespScaleRMSE", "CorrRMSE", "PRA", "SimUniqueRatio"])
    old = pd.read_csv(ANALYSIS / "coldstart_baselines.csv")
    old = old[old.method == "profile_center"]
    _check("profile_center oracle and cold-start, coldstart_baselines.csv", paper, old, ["dataset", "setting", "method"],
           ["RespItemMAE", "RespScaleRMSE", "ItemEMR"])

    if split_out:
        splits = pd.DataFrame([r for rows, _ in split_out for r in rows])
        splits.insert(4, "information", splits.method.map(INFORMATION))
        splits.to_csv(ANALYSIS / "r2_baselines_splits.csv", index=False)
        diag += [d for _, d in split_out]
        summary = splits.groupby(["dataset", "setting", "method"])[show].agg(["mean", "std", "min", "max"]).round(6)
        summary.to_csv(ANALYSIS / "r2_baselines_splits_summary.csv")
        old = pd.read_csv(ANALYSIS / "split_robustness.csv")
        _check("fifty splits, oracle, split_robustness.csv", splits[splits.setting == "oracle"], old,
               ["dataset", "seed", "method"], [k for k in METRIC_KEYS if k in old.columns])

        print(f"\n=== {args.splits} random splits, mean (SD) across splits ===")
        for (ds, setting), g in splits.groupby(["dataset", "setting"], sort=False):
            mean = g.groupby("method")[show].mean().loc[ORDER]
            sd = g.groupby("method")[show].std().loc[ORDER]
            table = mean.round(3).astype(str) + " (" + sd.round(3).astype(str) + ")"
            print(f"\n{ds}, {setting} assignment")
            print(table.to_string())
        # paired comparison with the profile-center baseline, split by split
        print("\n=== item MAE against profile_center, paired by split ===")
        rows = []
        for (ds, setting), g in splits.groupby(["dataset", "setting"], sort=False):
            wide = g.pivot(index="seed", columns="method", values="RespItemMAE")
            for name in ORDER:
                if name == "profile_center":
                    continue
                diff = wide[name] - wide["profile_center"]
                rows.append({"dataset": ds, "setting": setting, "method": name, "mean_difference": diff.mean(),
                             "sd_difference": diff.std(), "splits_better_than_profile_center": int((diff < 0).sum()),
                             "splits": len(diff)})
        paired = pd.DataFrame(rows)
        paired.round(6).to_csv(ANALYSIS / "r2_baselines_paired.csv", index=False)
        print(paired.round(3).to_string(index=False))

    # the ten splits of the LLM arms: every baseline against every arm, split by split
    base = pd.DataFrame([r for rows, _ in llm_out for r in rows])
    base = base[base.setting == "oracle"]                   # the repeated LLM runs use oracle assignment
    base.insert(4, "information", base.method.map(INFORMATION))
    base.to_csv(ANALYSIS / "r2_baselines_llm_splits.csv", index=False)
    diag += [d for _, d in llm_out]
    matched = [d["matches_llm_run"] for _, d in llm_out]
    print(f"\n=== the ten LLM splits ===\nheld-out respondents and profile partition identical to the stored LLM runs: {sum(matched)} of {len(matched)}")
    pairs = {"RespItemMAE": "respondent_item_mae_mean", "CorrRMSE": "correlation_rmse",
             "PRA": "profile_recovery_accuracy", "SimUniqueRatio": "simulated_unique_ratio"}
    rows = []
    for (ds, model, arm), g in llm_runs.groupby(["dataset", "model", "arm"]):
        g = g.set_index("seed")
        for name in ORDER:
            b = base[(base.dataset == ds) & (base.method == name)].set_index("seed").loc[g.index]
            row = {"dataset": ds, "model": model, "arm": arm, "baseline": name, "splits": len(g)}
            for ours, theirs in pairs.items():
                diff = g[theirs] - b[ours]
                row[f"{ours}_llm"] = g[theirs].mean()
                row[f"{ours}_baseline"] = b[ours].mean()
                row[f"{ours}_difference"] = diff.mean()
                better = diff > 0 if ours in ("PRA", "SimUniqueRatio") else diff < 0
                row[f"{ours}_splits_llm_better"] = int(better.sum())
            rows.append(row)
    versus = pd.DataFrame(rows)
    versus.round(6).to_csv(ANALYSIS / "r2_baselines_vs_llm.csv", index=False)
    for metric, title in (("RespItemMAE", "item MAE"), ("CorrRMSE", "correlation RMSE")):
        print(f"\n{title}: number of the ten splits in which the LLM arm is better than the baseline (GPT-5.4, all arms)")
        g = versus[versus.model == "GPT-5.4"]
        for ds in DATA:
            print(ds)
            print(g[g.dataset == ds].pivot(index="arm", columns="baseline", values=f"{metric}_splits_llm_better")[ORDER]
                  .rename(columns=lambda c: c.replace("ordreg_", "or_").replace("profile_metadata", "prof_meta")
                          .replace("population", "pop")).to_string())
    print("\nmean over the ten splits, baselines")
    print(base.groupby(["dataset", "method"])[["RespItemMAE", "CorrRMSE", "PRA", "SimUniqueRatio"]].mean()
          .unstack(0).loc[ORDER].round(3).to_string())
    print("\nmean over the ten splits, LLM arms (item MAE and correlation RMSE)")
    print(llm_runs.pivot_table(index=["dataset", "arm"], columns="model",
                               values="respondent_item_mae_mean").round(3).to_string())
    print(llm_runs.pivot_table(index=["dataset", "arm"], columns="model", values="correlation_rmse").round(3).to_string())

    diag = pd.DataFrame(diag)
    if not split_out and (ANALYSIS / "r2_baselines_fit.csv").exists():      # keep the rows of the 50 splits
        old_fit = pd.read_csv(ANALYSIS / "r2_baselines_fit.csv")
        diag = pd.concat([diag, old_fit[old_fit.analysis == "splits"]], ignore_index=True)
    diag.to_csv(ANALYSIS / "r2_baselines_fit.csv", index=False)
    conv = [c for c in diag.columns if c.startswith("grm_converged")]
    print("\n=== model fitting ===")
    print(f"graded response models converged: {int(diag[conv].to_numpy().sum())} of {diag[conv].size}")
    print(f"copula correlation matrices needing repair (smallest eigenvalue below 1e-6): "
          f"population {int((diag.copula_population_min_eigenvalue < 1e-6).sum())}, "
          f"profile {int((diag.copula_profile_min_eigenvalue < 1e-6).sum())} of {len(diag)}")
    print(f"largest share of correlations at the bound: {diag[['copula_population_share_at_bound', 'copula_profile_share_at_bound']].to_numpy().max():.3f}")
    print("ridge penalty chosen by cross-validation:")
    print(diag.groupby("dataset")[["ordreg_metadata_penalty", "ordreg_profile_metadata_penalty"]].agg(lambda s: dict(s.value_counts().sort_index())).to_string())
    print(f"\nwrote r2_baselines_paper.csv ({len(paper)} rows), r2_baselines_fit.csv ({len(diag)})"
          + ("" if not split_out else f", r2_baselines_splits.csv ({len(splits)}), r2_baselines_splits_summary.csv, r2_baselines_paired.csv"))


if __name__ == "__main__":
    main()
