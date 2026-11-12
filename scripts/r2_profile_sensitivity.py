"""Sensitivity of the benchmark to how the profiles are built (Reviewer 1, comment 3).

The profile layer is a Gaussian mixture with full covariance, fitted on the training
split's scale means, and every respondent is given one profile. This script measures
how much depends on those choices, in five parts.

A  agreement   Alternative partitions of the same training data (k-means, Ward,
               mixtures with diagonal and tied covariance, two to five profiles) and
               their adjusted Rand index against the partition the pipeline selects.
B  uncertainty How sure the mixture is of a respondent's profile: largest posterior
               probability, share of respondents below 0.7, and posterior entropy.
C  baselines   The two profile-dependent baselines recomputed under every alternative
               partition.
D  soft        A profile-mean baseline that weights the profile means by membership
               probabilities, against the hard-assignment version. Under oracle
               assignment the probabilities are the mixture's posteriors, under
               cold-start the metadata classifier's (the training base rates when
               cross-validation selects the constant classifier).
E  rescoring   Profile recovery and within-profile error of the existing LLM records,
               recomputed with the held-out respondents labelled by every alternative
               partition, next to the majority rate in the same label space.

Parts A to D use the 50 random splits of ``split_robustness.py``. Part E uses the LLM
runs of the paper's split (final test) and of the ten repeated splits. In part E the
records were generated from persona cards built on the pipeline's own partition, so it
shows how the metric depends on the partition; whether generation depends on it is a
separate question that needs new runs.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_profile_sensitivity.py [--workers 10]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from compute_nonllm_baselines import (  # noqa: E402
    _columns, _dataset_settings, _evaluate, _impute_train_items, _item_bounds, _recompute_scale_means,
    baseline_profile_center, baseline_within_profile,
)
from saykopersona.coldstart import expand_multi_select, fit_coldstart_predictor  # noqa: E402
from saykopersona.data import build_stratified_three_way_split, load_raw_survey, score_survey_dataframe  # noqa: E402
from saykopersona.profiles import fit_empirical_profiles, predict_profiles  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "results" / "analysis"
RUNS = REPO / "results" / "runs" / "experiment"
DATA = {"anket_tr": REPO / "datasets" / "anket.xlsx",
        "bangladesh_sns": REPO / "datasets" / "bangladesh_sns" / "mmc2.xlsx"}
FEATURES = {"anket_tr": ["digital_addiction_mean", "social_anxiety_mean", "wellbeing_mean"],
            "bangladesh_sns": ["ucla_loneliness_mean", "phq9_mean", "gad7_mean"]}
METHODS = ["gmm_full", "gmm_diag", "gmm_tied", "kmeans", "ward"]
COUNTS = [2, 3, 4, 5]
REFERENCE = "pipeline"
SEED = 20261112


# --------------------------------------------------------------------------- #
def alternative_partitions(train_z: np.ndarray, test_z: np.ndarray, seed: int) -> dict:
    """(method, count) -> (training labels, held-out labels), features already standardised."""
    out = {}
    for k in COUNTS:
        for cov in ("full", "diag", "tied"):
            gm = GaussianMixture(n_components=k, covariance_type=cov, random_state=seed, n_init=10,
                                 reg_covar=1e-6).fit(train_z)
            out[(f"gmm_{cov}", k)] = (gm.predict(train_z), gm.predict(test_z))
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(train_z)
        out[("kmeans", k)] = (km.labels_, km.predict(test_z))
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(train_z)
        centers = np.vstack([train_z[ward.labels_ == c].mean(axis=0) for c in range(k)])
        nearest = ((test_z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        out[("ward", k)] = (ward.labels_, nearest)      # Ward has no prediction rule; nearest centre is used
    return out


def recovery_labels(human: pd.DataFrame, frame: pd.DataFrame, scales: list[str]) -> np.ndarray:
    """Nearest standardised centroid, the label space of the profile-recovery metric."""
    centroids = human.groupby("profile_id")[scales].mean()
    spread = human[scales].std(ddof=1).replace(0.0, 1.0)
    c = (centroids / spread).to_numpy()
    z = (frame[scales] / spread).to_numpy()
    return centroids.index.to_numpy()[((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]


def soft_center(weights: np.ndarray, labels: np.ndarray, frame: pd.DataFrame, train: pd.DataFrame, cols) -> pd.DataFrame:
    """Profile means weighted by membership probabilities, rounded and clipped like the hard version."""
    centers = train.groupby("profile_id")[cols.item_columns].mean().loc[labels].to_numpy()
    bounds = _item_bounds(train, cols)
    lo = np.array([bounds[i][0] for i in cols.item_columns])
    hi = np.array([bounds[i][1] for i in cols.item_columns])
    out = frame[["respondent_id", "profile_id"]].reset_index(drop=True).copy()
    out[cols.item_columns] = np.clip(np.round(weights @ centers), lo, hi)
    return _recompute_scale_means(out, cols)


def uncertainty(prob: np.ndarray) -> dict:
    top = prob.max(axis=1)
    entropy = -(prob * np.log(np.clip(prob, 1e-300, None))).sum(axis=1) / np.log(prob.shape[1])
    return {"mean_max_posterior": float(top.mean()), "share_below_0.7": float((top < 0.7).mean()),
            "share_below_0.9": float((top < 0.9).mean()), "mean_entropy": float(entropy.mean())}


# --------------------------------------------------------------------------- #
# parts A to D: one random split                                               #
# --------------------------------------------------------------------------- #
def one_split(task: tuple[str, int]) -> dict:
    warnings.filterwarnings("ignore")
    dataset, seed = task
    settings = _dataset_settings(dataset)
    cols = _columns(settings)
    feats = FEATURES[dataset]
    scored = score_survey_dataframe(load_raw_survey(DATA[dataset], dataset=settings), dataset=settings)
    train, _val, test = build_stratified_three_way_split(
        scored, validation_size=0.1, final_test_size=0.1, random_state=seed, dataset=settings)
    train = train.dropna(subset=feats)
    test = test.dropna(subset=feats)
    result = fit_empirical_profiles(train, feature_columns=feats, random_state=seed,
                                    anchor_columns=list(settings.profile_anchor_columns))
    ref_train = result.assignments["profile_id"].to_numpy()
    ref_test = predict_profiles(result, test).to_numpy()
    train_z = result.scaler.transform(train[feats].astype(float))
    test_z = result.scaler.transform(test[feats].astype(float))
    key = {"dataset": dataset, "seed": seed, "pipeline_count": result.selected_k}

    train_i = _recompute_scale_means(_impute_train_items(train.copy(), cols), cols)
    test_i = _recompute_scale_means(test.copy(), cols)

    agreement, baselines = [], []
    parts = {(REFERENCE, result.selected_k): (ref_train, ref_test), **alternative_partitions(train_z, test_z, seed)}
    for m, ((method, k), (lab_train, lab_test)) in enumerate(parts.items()):
        row = {**key, "method": method, "count": k, "same_count": k == result.selected_k,
               "ari_train": adjusted_rand_score(ref_train, lab_train),
               "ari_test": adjusted_rand_score(ref_test, lab_test),
               "silhouette": float(silhouette_score(train_z, lab_train)),
               "smallest_profile": int(np.bincount(lab_train).min()),
               "empty_in_test": int(k - len(np.unique(lab_test)))}
        agreement.append(row)
        tr = train_i.assign(profile_id=lab_train)
        te = test_i.assign(profile_id=lab_test)
        rng = np.random.default_rng([SEED, seed, m])
        for name, sim in (("profile_center", baseline_profile_center(te, tr, cols)),
                          ("within_profile", baseline_within_profile(te, tr, cols, rng))):
            baselines.append({**key, "method": method, "count": k, "same_count": k == result.selected_k,
                              "baseline": name, **_evaluate(te, sim, cols)})

    # B: assignment uncertainty of the pipeline's mixture
    unc = [{**key, "sample": "training", **uncertainty(result.model.predict_proba(train_z))},
           {**key, "sample": "held-out", **uncertainty(result.model.predict_proba(test_z))}]

    # D: soft against hard profile means
    tr = train_i.assign(profile_id=ref_train)
    soft = []
    post = result.model.predict_proba(test_z)
    te = test_i.assign(profile_id=ref_test)
    soft.append({**key, "setting": "oracle", "version": "hard", **_evaluate(te, baseline_profile_center(te, tr, cols), cols)})
    soft.append({**key, "setting": "oracle", "version": "soft",
                 **_evaluate(te, soft_center(post, np.arange(post.shape[1]), te, tr, cols), cols)})
    predictor = fit_coldstart_predictor(result.assignments, dataset=settings, random_state=seed)
    frame = test.copy()
    if predictor.multi_select:
        frame, _ = expand_multi_select(frame, predictor.multi_select)
        for column in predictor.numeric:
            if column not in frame.columns:
                frame[column] = 0
    prob = predictor.model.predict_proba(frame)
    classes = np.asarray(predictor.model.classes_).astype(int)
    te = test_i.assign(profile_id=classes[prob.argmax(axis=1)])
    if predictor.selected == "majority":
        # The constant classifier reports certainty for the largest profile. Its honest
        # membership probabilities are the training base rates, so those are the weights.
        rates = pd.Series(ref_train).value_counts(normalize=True).reindex(classes).to_numpy()
        prob = np.tile(rates, (len(frame), 1))
    extra = {"coldstart_model": predictor.selected, **{f"coldstart_{k}": v for k, v in uncertainty(prob).items()}} \
        if prob.shape[1] > 1 else {"coldstart_model": predictor.selected}
    soft.append({**key, "setting": "coldstart", "version": "hard", **extra,
                 **_evaluate(te, baseline_profile_center(te, tr, cols), cols)})
    soft.append({**key, "setting": "coldstart", "version": "soft", **extra,
                 **_evaluate(te, soft_center(prob, classes, te, tr, cols), cols)})
    return {"agreement": agreement, "baselines": baselines, "uncertainty": unc, "soft": soft}


# --------------------------------------------------------------------------- #
# part E: existing LLM records under alternative partitions                    #
# --------------------------------------------------------------------------- #
def rescore_group(task: tuple[str, str, int, list[tuple[str, str, str]]]) -> list[dict]:
    warnings.filterwarnings("ignore")
    dataset, source, seed, runs = task
    cols = _columns(_dataset_settings(dataset))
    feats = FEATURES[dataset]
    out, parts, shared = [], None, None
    for model, arm, run in runs:
        d = RUNS / run
        split = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))["evaluation_split"]
        if parts is None:                    # the data and profiles are the same for every run of a split
            train = pd.read_csv(d / "data" / "train_split.csv").merge(
                pd.read_csv(d / "profiles" / "train_profile_assignments.csv")[["respondent_id", "profile_id"]],
                on="respondent_id").dropna(subset=feats)
            human = pd.read_csv(d / "data" / f"{split}_split.csv").merge(
                pd.read_csv(d / "profiles" / f"{split}_profile_assignments.csv")[["respondent_id", "profile_id"]],
                on="respondent_id")
            human = _recompute_scale_means(human, cols)
            mean, std = train[feats].mean().to_numpy(), train[feats].std(ddof=0).to_numpy()
            train_z = (train[feats].to_numpy(dtype=float) - mean) / std
            test_z = (human[feats].to_numpy(dtype=float) - mean) / std
            ref_count = int(train.profile_id.nunique())
            parts = {(REFERENCE, ref_count): (train.profile_id.to_numpy(), human.profile_id.to_numpy()),
                     **alternative_partitions(train_z, test_z, seed)}
            refit = adjusted_rand_score(train.profile_id, parts[("gmm_full", ref_count)][0])
            shared = (human, ref_count, refit)
        human, ref_count, refit = shared
        sim = pd.read_csv(d / "simulation" / "simulated_responses.csv").set_index("respondent_id")
        sim = _recompute_scale_means(sim.loc[human.respondent_id].reset_index(), cols)
        for (method, k), (_, lab_test) in parts.items():
            h = human.assign(profile_id=lab_test)
            s = sim.assign(profile_id=lab_test)
            ev = _evaluate(h, s, cols)
            if len(np.unique(lab_test)) > 1:
                hl = recovery_labels(h, h, cols.scale_columns)
                sl = recovery_labels(h, s, cols.scale_columns)
                own, majority = float((hl == sl).mean()), float(pd.Series(hl).value_counts(normalize=True).max())
            else:
                own, majority = float("nan"), float("nan")
            out.append({"dataset": dataset, "source": source, "seed": seed, "model": model, "arm": arm, "run": run,
                        "method": method, "count": k, "pipeline_count": ref_count, "same_count": k == ref_count,
                        "refit_ari": refit, "recovery": ev["PRA"], "recovery_check": own, "majority": majority,
                        "within_rmse": ev["WithinRMSE"], "item_mae": ev["RespItemMAE"]})
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    idx = pd.read_csv(ANALYSIS / "run_index.csv")
    idx = idx[idx.split == "final_test"].assign(seed=42, source="paper_split")
    rep = pd.read_csv(ANALYSIS / "c2_llm_runs.csv").assign(source="repeated_splits")
    runs = pd.concat([idx[["dataset", "source", "seed", "model", "arm", "run"]],
                      rep[["dataset", "source", "seed", "model", "arm", "run"]]], ignore_index=True)
    groups = [(ds, src, int(seed), list(g[["model", "arm", "run"]].itertuples(index=False, name=None)))
              for (ds, src, seed), g in runs.groupby(["dataset", "source", "seed"])]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        split_jobs = [pool.submit(one_split, (ds, s)) for ds in DATA for s in range(args.splits)]
        score_jobs = [pool.submit(rescore_group, g) for g in groups]
        split_out = [j.result() for j in split_jobs]
        score_out = [j.result() for j in score_jobs]

    agreement = pd.DataFrame([r for o in split_out for r in o["agreement"]])
    baselines = pd.DataFrame([r for o in split_out for r in o["baselines"]])
    unc = pd.DataFrame([r for o in split_out for r in o["uncertainty"]])
    soft = pd.DataFrame([r for o in split_out for r in o["soft"]])
    rescored = pd.DataFrame([r for o in score_out for r in o])
    agreement.to_csv(ANALYSIS / "r2_profile_agreement.csv", index=False)
    baselines.to_csv(ANALYSIS / "r2_profile_baselines.csv", index=False)
    unc.to_csv(ANALYSIS / "r2_profile_uncertainty.csv", index=False)
    soft.to_csv(ANALYSIS / "r2_profile_soft_baseline.csv", index=False)
    rescored.to_csv(ANALYSIS / "r2_profile_rescoring.csv", index=False)

    print("=== checks ===")
    same = agreement[(agreement.method == "gmm_full") & agreement.same_count]
    print(f"refitting the pipeline's mixture at its own count reproduces its partition: "
          f"ARI {same.ari_train.min():.3f} to {same.ari_train.max():.3f} over {len(same)} splits")
    ref = rescored[rescored.method == REFERENCE]
    print(f"stored run partitions reproduced by refitting: ARI {ref.refit_ari.min():.3f} to {ref.refit_ari.max():.3f} "
          f"over {ref.groupby(['dataset', 'source', 'seed']).ngroups} splits")
    ok = rescored.dropna(subset=["recovery_check"])
    print(f"recovery recomputed in this script against the evaluation code: largest difference "
          f"{(ok.recovery - ok.recovery_check).abs().max():.2e} over {len(ok)} rows")
    reps = pd.read_csv(ANALYSIS / "c2_llm_runs.csv").merge(ref, on=["dataset", "model", "arm", "seed", "run"])
    print(f"reference-partition recovery against c2_llm_runs.csv: largest difference "
          f"{(reps.profile_recovery_accuracy - reps.recovery).abs().max():.2e} over {len(reps)} runs")

    print(f"\n=== A. pipeline's selected profile count over {args.splits} splits ===")
    print(agreement[agreement.method == REFERENCE].groupby("dataset")["count"].value_counts().unstack(fill_value=0).to_string())
    print("\n=== A. adjusted Rand index against the pipeline's partition, alternatives at the same count ===")
    a = agreement[agreement.same_count & (agreement.method != REFERENCE)]
    print(a.groupby(["dataset", "method"])[["ari_train", "ari_test", "silhouette", "smallest_profile"]]
          .agg(["mean", "min"]).round(3).to_string())
    print("\n=== A. adjusted Rand index on the training split, by method and count (mean over splits) ===")
    print(agreement[agreement.method != REFERENCE].pivot_table(index=["dataset", "method"], columns="count",
                                                               values="ari_train").round(3).to_string())

    print("\n=== B. assignment uncertainty of the pipeline's mixture (mean over splits, range) ===")
    print(unc.groupby(["dataset", "sample"])[["mean_max_posterior", "share_below_0.7", "share_below_0.9", "mean_entropy"]]
          .agg(["mean", "min", "max"]).round(3).to_string())

    print("\n=== C. profile-dependent baselines under alternative partitions (mean over splits) ===")
    for name in ("profile_center", "within_profile"):
        b = baselines[baselines.baseline == name]
        ref_b = b[b.method == REFERENCE].groupby("dataset")[["RespItemMAE", "PRA"]].mean()
        print(f"\n{name}; pipeline partition: " + "; ".join(f"{d} item MAE {r.RespItemMAE:.3f}, recovery {r.PRA:.3f}"
                                                              for d, r in ref_b.iterrows()))
        print("item MAE")
        print(b[b.method != REFERENCE].pivot_table(index=["dataset", "method"], columns="count", values="RespItemMAE").round(3).to_string())
        print("recovery")
        print(b[b.method != REFERENCE].pivot_table(index=["dataset", "method"], columns="count", values="PRA").round(3).to_string())

    print("\n=== D. soft against hard profile means (paired by split) ===")
    rows = []
    for (ds, setting), g in soft.groupby(["dataset", "setting"], sort=False):
        w = g.pivot(index="seed", columns="version")
        for metric in ("RespItemMAE", "RespScaleRMSE", "ItemEMR", "SimUniqueRatio"):
            diff = w[(metric, "soft")] - w[(metric, "hard")]
            rows.append({"dataset": ds, "setting": setting, "metric": metric, "hard": w[(metric, "hard")].mean(),
                         "soft": w[(metric, "soft")].mean(), "difference": diff.mean(), "sd_difference": diff.std(),
                         "splits_soft_lower": int((diff < 0).sum()), "splits_equal": int((diff == 0).sum()), "splits": len(diff)})
    paired = pd.DataFrame(rows)
    cold = soft[soft.setting == "coldstart"].pivot(index=["dataset", "seed", "coldstart_model"], columns="version",
                                                   values="RespItemMAE").reset_index()
    cold["constant_classifier"] = cold.coldstart_model == "majority"
    cold["difference"] = cold.soft - cold.hard
    by_model = cold.groupby(["dataset", "constant_classifier"]).agg(
        splits=("difference", "size"), hard=("hard", "mean"), soft=("soft", "mean"), difference=("difference", "mean"),
        sd_difference=("difference", "std"), splits_soft_lower=("difference", lambda d: int((d < 0).sum())))
    by_model.round(6).to_csv(ANALYSIS / "r2_profile_soft_by_classifier.csv")
    paired.round(6).to_csv(ANALYSIS / "r2_profile_soft_paired.csv", index=False)
    print(paired.round(3).to_string(index=False))
    print("cold-start item MAE, by whether cross-validation selected the constant classifier:")
    print(by_model.round(3).to_string())
    cs = soft[(soft.setting == "coldstart") & (soft.version == "hard")]
    print("cold-start classifier selected per split:")
    print(cs.groupby("dataset").coldstart_model.value_counts().unstack(fill_value=0).to_string())

    print("\n=== E. LLM records re-scored under alternative partitions ===")
    arms = ["DOP", "BPP", "SADP", "HC"]
    r = rescored[rescored.arm.isin(arms)]
    for source in ("paper_split", "repeated_splits"):
        sub = r[(r.source == source) & (r.same_count)]
        print(f"\n{source}: recovery by arm at the pipeline's count, mean over model families"
              + (" and splits" if source == "repeated_splits" else ""))
        t = sub.pivot_table(index=["dataset", "method"], columns="arm", values="recovery")[arms]
        t["majority"] = sub.groupby(["dataset", "method"]).majority.mean()
        print(t.round(3).to_string())
    # contrasts cell by cell: one cell is a dataset, split, model family, method and count
    wide = r.pivot_table(index=["dataset", "source", "seed", "model", "method", "count", "same_count"],
                         columns="arm", values="recovery").reset_index()
    wide["majority"] = r.groupby(["dataset", "source", "seed", "model", "method", "count", "same_count"]).majority.first().values
    wide = wide.dropna(subset=arms)
    contrasts = {"BPP above DOP": wide.BPP > wide.DOP, "SADP above DOP": wide.SADP > wide.DOP,
                 "SADP at or above BPP": wide.SADP >= wide.BPP, "HC below SADP": wide.HC < wide.SADP,
                 "DOP at or below majority": wide.DOP <= wide.majority}
    for name, flag in contrasts.items():
        wide[name] = flag
    summary = wide.groupby(["dataset", "method", "count"])[list(contrasts)].mean()
    summary["cells"] = wide.groupby(["dataset", "method", "count"]).size()
    summary.round(6).to_csv(ANALYSIS / "r2_profile_rescoring_contrasts.csv")
    print("\nshare of cells (model family by split) in which each recovery contrast holds, repeated and paper splits pooled")
    print("pipeline partition:")
    print(summary.xs(REFERENCE, level="method").round(3).to_string())
    print("alternative partitions, by count (all four methods pooled):")
    alt = wide[wide.method != REFERENCE]
    print(alt.groupby(["dataset", "count"])[list(contrasts)].mean().round(3).to_string())
    print("alternative partitions, by method (all counts pooled):")
    print(alt.groupby(["dataset", "method"])[list(contrasts)].mean().round(3).to_string())

    print("\n=== E. arm contrasts by the pipeline's own profile count (repeated splits, pipeline partition) ===")
    p = wide[(wide.method == REFERENCE) & (wide.source == "repeated_splits")]
    print(p.groupby(["dataset", "count"]).agg(cells=("BPP", "size"), splits=("seed", "nunique"), DOP=("DOP", "mean"),
                                              BPP=("BPP", "mean"), SADP=("SADP", "mean"), HC=("HC", "mean"),
                                              majority=("majority", "mean"),
                                              **{k: (k, "mean") for k in contrasts}).round(3).to_string())
    print(f"\nwrote r2_profile_agreement.csv ({len(agreement)}), r2_profile_baselines.csv ({len(baselines)}), "
          f"r2_profile_uncertainty.csv ({len(unc)}), r2_profile_soft_baseline.csv ({len(soft)}), "
          f"r2_profile_soft_paired.csv, r2_profile_rescoring.csv ({len(rescored)}), r2_profile_rescoring_contrasts.csv")


if __name__ == "__main__":
    main()
