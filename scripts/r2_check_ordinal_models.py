"""Checks of the estimators in ``r2_ordinal_models.py`` against reference software.

1. Bivariate normal CDF against scipy's multivariate normal CDF.
2. Graded response model item parameters against mirt (R), same data.
3. Polychoric correlations against polycor::polychor (R, two-step), same data.
4. Proportional-odds regression against statsmodels OrderedModel without a penalty, and
   the analytic gradient against finite differences.
5. Recovery of a non-normal latent distribution by the empirical-histogram estimator
   on simulated data.

The data are the training split of the paper's Turkish and Bangladesh runs.

Usage
-----
    PYTHONPATH=src:scripts python scripts/r2_check_ordinal_models.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.stats import multivariate_normal

import r2_ordinal_models as M
from compute_nonllm_baselines import _columns, _dataset_settings, _impute_train_items, _load_run

REPO = Path(__file__).resolve().parents[1]
RUNS = {"anket_tr": "20260907T074129231220Z-d5aa036f", "bangladesh_sns": "20260908T113102121289Z-80bd6559"}
RSCRIPT = "/opt/homebrew/bin/Rscript"
R_CODE = r"""
suppressMessages({library(mirt); library(polycor)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
mod <- mirt(d, 1, itemtype = "graded", verbose = FALSE, TOL = 1e-5, quadpts = 61)
write.csv(coef(mod, simplify = TRUE)$items, args[2])
n <- ncol(d); out <- matrix(1, n, n)
for (i in 1:(n - 1)) for (j in (i + 1):n) { r <- polychor(d[[i]], d[[j]], ML = FALSE); out[i, j] <- r; out[j, i] <- r }
write.csv(out, args[3], row.names = FALSE)
"""


def check_bvn(rng: np.random.Generator) -> float:
    worst = 0.0
    for _ in range(400):
        h, k = rng.uniform(-4, 4, 2)
        rho = rng.choice([rng.uniform(-0.97, 0.97), 0.97, -0.97, 0.9])
        ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
        worst = max(worst, abs(float(M.bvn_cdf(np.array(h), np.array(k), np.array(rho))) - ref))
    return worst


def main() -> None:
    rng = np.random.default_rng(20261112)
    print(f"1. bivariate normal CDF, largest absolute difference from scipy over 400 points: {check_bvn(rng):.2e}")

    for ds, run in RUNS.items():
        cols = _columns(_dataset_settings(ds))
        _, train = _load_run(REPO / "results" / "runs" / "experiment" / run, cols)
        train = _impute_train_items(train, cols)
        scale, items = next(iter(cols.scale_to_items.items()))
        raw = train[items].astype(float).to_numpy()
        cats = [np.unique(raw[:, j]) for j in range(raw.shape[1])]
        x = M.encode_items(raw, cats)
        ks = [len(c) for c in cats]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.DataFrame(x, columns=[f"i{j}" for j in range(x.shape[1])]).to_csv(tmp / "x.csv", index=False)
            (tmp / "check.R").write_text(R_CODE)
            subprocess.run([RSCRIPT, str(tmp / "check.R"), str(tmp / "x.csv"), str(tmp / "grm.csv"),
                            str(tmp / "poly.csv")], check=True, capture_output=True)
            ref_grm = pd.read_csv(tmp / "grm.csv", index_col=0)
            ref_poly = pd.read_csv(tmp / "poly.csv").to_numpy()

        grm = M.GradedResponse(tol=1e-5).fit(x, ks)
        ref_a = ref_grm["a1"].to_numpy()
        ref_d = ref_grm[[c for c in ref_grm.columns if c.startswith("d")]].to_numpy()
        d = np.vstack(grm.d_)
        print(f"\n{ds}, scale {scale} ({x.shape[1]} items, n = {len(x)})")
        print(f"2. graded response model: converged {grm.converged_} in {grm.cycles_} cycles, log-likelihood {grm.loglik_:.3f}")
        print(f"   slopes      largest |difference| from mirt {np.abs(grm.a_ - ref_a).max():.4f} (slopes range {ref_a.min():.2f} to {ref_a.max():.2f})")
        print(f"   intercepts  largest |difference| from mirt {np.abs(d - ref_d).max():.4f}")

        cop = M.OrdinalCopula().fit(x, np.zeros(len(x), dtype=int), ks)
        off = ~np.eye(x.shape[1], dtype=bool)
        print(f"3. polychoric correlations: largest |difference| from polycor {np.abs(cop.rho_raw_ - ref_poly)[off].max():.5f} "
              f"(values range {ref_poly[off].min():.2f} to {ref_poly[off].max():.2f})")

        from statsmodels.miscmodels.ordinal_model import OrderedModel
        covs = pd.get_dummies(train[["gender", "age_band"]].astype(str), drop_first=True).to_numpy(dtype=float)
        covs = (covs - covs.mean(axis=0)) / covs.std(axis=0)
        y = x[:, 0]
        beta, alpha = M.po_fit(covs, y, ks[0], 0.0)
        ref = OrderedModel(y, covs, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
        ref_beta = ref.params[:covs.shape[1]]
        ref_alpha = ref.model.transform_threshold_params(ref.params[covs.shape[1]:])[1:-1]
        start = rng.normal(0, 0.3, covs.shape[1] + ks[0] - 1)
        grad_err = check_grad(lambda p: M.po_objective(p, covs, y, ks[0], 2.0)[0],
                              lambda p: M.po_objective(p, covs, y, ks[0], 2.0)[1], start)
        print(f"4. proportional odds: largest |difference| from statsmodels, coefficients {np.abs(beta - ref_beta).max():.5f}, "
              f"thresholds {np.abs(alpha - ref_alpha).max():.5f}; gradient error {grad_err:.2e}")

    # 5. recovery of a non-normal latent distribution from simulated answers: two
    #    clusters, one of them at the floor of the third scale, as in a low-symptom profile
    n = 3000
    low = rng.random(n) < 0.4
    theta = np.where(low[:, None], rng.multivariate_normal([-1.0, -1.2, -2.6], np.diag([0.5, 0.3, 0.05]), n),
                     rng.multivariate_normal([0.6, 0.4, 0.5], [[0.5, 0.3, 0.2], [0.3, 0.6, 0.3], [0.2, 0.3, 0.4]], n))
    grid = np.linspace(-4.0, 4.0, 41)
    logliks, answers, params = [], [], []
    for s in range(3):
        g = M.GradedResponse()
        g.a_ = rng.uniform(1.5, 3.5, 9)
        g.d_ = [np.sort(rng.uniform(-3.0, 3.0, 3))[::-1] for _ in range(9)]
        params.append((g.a_, g.d_))
        answers.append(g.sample(theta[:, s], rng))
        logliks.append(g.loglik_on(answers[-1], grid))
    pi, cycles, _ = M.latent_histogram(logliks, grid)
    models = []
    for s in range(3):
        g = M.GradedResponse()
        g.a_, g.d_ = params[s]
        models.append(g)
    sim_theta = M.sample_latent(pi, grid, 60000, rng)
    regenerated = [models[s].sample(sim_theta[:, s], rng) for s in range(3)]
    obs = np.column_stack([a.mean(axis=1) for a in answers])
    reg = np.column_stack([a.mean(axis=1) for a in regenerated])
    tri = np.triu_indices(3, 1)
    print(f"\n5. empirical-histogram latent distribution from {n} simulated respondents, {cycles} cycles")
    print("   scale scores of the simulated data against data regenerated from the estimate")
    print(f"   means         simulated {obs.mean(axis=0).round(3)}  regenerated {reg.mean(axis=0).round(3)}")
    print(f"   SDs           simulated {obs.std(axis=0).round(3)}  regenerated {reg.std(axis=0).round(3)}")
    print(f"   correlations  simulated {np.corrcoef(obs.T)[tri].round(3)}  regenerated {np.corrcoef(reg.T)[tri].round(3)}")
    print(f"   share answering all zeros on the third scale: simulated {(answers[2].sum(axis=1) == 0).mean():.3f}, "
          f"regenerated {(regenerated[2].sum(axis=1) == 0).mean():.3f}")

if __name__ == "__main__":
    main()
