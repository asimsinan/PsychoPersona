"""Conventional generators of ordinal questionnaire data (Reviewer 1, comment 2).

Three model classes, each fitted on the training split only.

GradedResponse   Samejima's graded response model, one latent trait per scale, estimated
                 by marginal maximum likelihood (EM, 61 quadrature nodes on [-6, 6]).
                 The joint distribution of the scale traits is then estimated for every
                 group of respondents (a profile, or the whole training split) without a
                 distributional assumption, as probabilities on a three-dimensional grid
                 (empirical histogram), with the item parameters held fixed. A synthetic
                 respondent draws traits from the group's distribution and item answers
                 from the model.
OrdinalCopula    Gaussian copula for ordinal items (the underlying-variable model). Each
                 item is a normal variable cut at thresholds; thresholds are taken from
                 the group's own answer proportions and the latent correlation of every
                 item pair is the two-step polychoric estimate. The correlation matrix is
                 common to all groups and estimated from all of them, because the
                 smallest profile holds about 50 training respondents, too few for the
                 several hundred correlations a per-profile matrix needs.
OrdinalRegression  Proportional-odds logistic regression of every item on respondent
                 covariates, with a ridge penalty chosen by five-fold cross-validated
                 log-likelihood on the training split. Answers are sampled from the
                 predicted category probabilities, or set to the predicted median, which
                 minimises expected absolute error.

``scripts/r2_check_ordinal_models.py`` checks the three estimators against mirt,
polycor and statsmodels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri

EPS = 1e-12


# --------------------------------------------------------------------------- #
# shared                                                                       #
# --------------------------------------------------------------------------- #
def encode_items(values: np.ndarray, categories: list[np.ndarray]) -> np.ndarray:
    """Map observed answers to 0-based category indices, item by item."""
    out = np.empty(values.shape, dtype=int)
    for j, cats in enumerate(categories):
        idx = np.searchsorted(cats, values[:, j])
        idx = np.clip(idx, 0, len(cats) - 1)
        if not np.array_equal(cats[idx], values[:, j]):
            raise ValueError(f"item {j} holds a value outside its training categories")
        out[:, j] = idx
    return out


def sample_categories(prob: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one category per row of a (rows x categories) probability matrix."""
    cum = np.cumsum(prob, axis=1)
    cum /= cum[:, -1:]
    u = rng.random(len(prob))
    return np.minimum((u[:, None] > cum).sum(axis=1), prob.shape[1] - 1)


# --------------------------------------------------------------------------- #
# graded response model                                                        #
# --------------------------------------------------------------------------- #
def grm_category_probs(a: float, d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """P(X = k | theta) for one item; ``d`` are the decreasing intercepts."""
    star = expit(a * theta[:, None] + d[None, :])
    upper = np.hstack([np.ones((len(theta), 1)), star])
    lower = np.hstack([star, np.zeros((len(theta), 1))])
    return np.clip(upper - lower, EPS, None)


def _unpack(par: np.ndarray) -> tuple[float, np.ndarray]:
    a = par[0]                # free in sign: a reverse-worded item can load negatively
    d = par[1] - np.concatenate([[0.0], np.cumsum(np.exp(par[2:]))])
    return a, d


def _pack(a: float, d: np.ndarray) -> np.ndarray:
    return np.concatenate([[a, d[0]], np.log(np.clip(-np.diff(d), 1e-6, None))])


class GradedResponse:
    """Unidimensional graded response model for the items of one scale."""

    nodes = np.linspace(-6.0, 6.0, 61)

    def __init__(self, max_cycles: int = 500, tol: float = 1e-4):
        self.max_cycles = max_cycles
        self.tol = tol

    def fit(self, x: np.ndarray, n_categories: list[int]) -> "GradedResponse":
        n, n_items = x.shape
        nodes = self.nodes
        prior = np.exp(-0.5 * nodes ** 2)
        prior /= prior.sum()
        onehot = [np.eye(k)[x[:, j]] for j, k in enumerate(n_categories)]

        pars = []
        total = x.sum(axis=1)
        for j, k in enumerate(n_categories):
            cum = np.clip(np.cumsum(np.bincount(x[:, j], minlength=k))[:-1] / n, 0.005, 0.995)
            rest = np.corrcoef(x[:, j], total - x[:, j])[0, 1] if n_items > 1 else 1.0
            pars.append(_pack(1.0 if not rest < 0 else -1.0, np.log((1 - cum) / cum)))

        self.loglik_ = -np.inf
        for cycle in range(1, self.max_cycles + 1):
            logl = np.zeros((n, len(nodes)))
            for j, par in enumerate(pars):
                logl += np.log(grm_category_probs(*_unpack(par), nodes))[:, x[:, j]].T
            joint = np.exp(logl) * prior
            marginal = joint.sum(axis=1)
            post = joint / marginal[:, None]
            self.loglik_ = float(np.log(marginal).sum())

            change = 0.0
            for j, par in enumerate(pars):
                r = post.T @ onehot[j]                                   # nodes x categories

                def nll(p, r=r):
                    return -float((r * np.log(grm_category_probs(*_unpack(p), nodes))).sum())

                new = minimize(nll, par, method="L-BFGS-B").x
                a0, d0 = _unpack(par)
                a1, d1 = _unpack(new)
                change = max(change, abs(a1 - a0), float(np.abs(d1 - d0).max()))
                pars[j] = new
            if change < self.tol:
                break
        self.cycles_ = cycle
        self.converged_ = change < self.tol
        self.a_ = np.array([_unpack(p)[0] for p in pars])
        self.d_ = [_unpack(p)[1] for p in pars]
        return self

    def loglik_on(self, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """log-likelihood of every respondent's answers at every grid value of the trait."""
        out = np.zeros((len(x), len(grid)))
        for j in range(x.shape[1]):
            out += np.log(grm_category_probs(self.a_[j], self.d_[j], grid))[:, x[:, j]].T
        return out

    def sample(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([sample_categories(grm_category_probs(self.a_[j], self.d_[j], theta), rng)
                                for j in range(len(self.a_))])


def latent_histogram(logliks: list[np.ndarray], grid: np.ndarray, max_cycles: int = 1000,
                     tol: float = 1e-3) -> tuple[np.ndarray, int, float]:
    """Nonparametric ML estimate of the joint distribution of three scale traits in a group.

    The distribution is a free set of probabilities on the three-dimensional grid (the
    empirical-histogram estimator), found by EM with the item parameters held fixed.
    ``logliks`` holds one (respondents x grid) log-likelihood array per scale. Nothing is
    assumed about the shape of the distribution or about independence between the scales.
    A normal distribution is not used within a profile because profiles are cut from the
    scale scores, and a profile at the floor of a scale has almost no trait variance.

    Returns the probabilities, the number of cycles and the log-likelihood. Iteration
    stops when a cycle raises the log-likelihood by less than ``tol``.
    """
    if len(logliks) != 3:
        raise ValueError("written for instruments with three scales")
    q = len(grid)
    n = len(logliks[0])
    shift = sum(float(ll.max(axis=1).sum()) for ll in logliks)
    l1, l2, l3 = [np.exp(ll - ll.max(axis=1, keepdims=True)) for ll in logliks]
    pair = (l1[:, :, None] * l2[:, None, :]).reshape(n, q * q)           # respondents x (q*q)
    pi = np.full((q * q, q), 1.0 / q ** 3)
    previous = -np.inf
    for cycle in range(1, max_cycles + 1):
        z = np.einsum("ip,ip->i", pair, l3 @ pi.T)                       # marginal likelihoods
        loglik = float(np.log(z).sum()) + shift
        pi = pi * ((pair / z[:, None]).T @ l3) / n
        if loglik - previous < tol:
            break
        previous = loglik
    return pi.reshape(q, q, q), cycle, loglik


def sample_latent(pi: np.ndarray, grid: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw trait vectors from a distribution on the grid."""
    flat = rng.choice(pi.size, size=size, p=pi.ravel() / pi.sum())
    return np.column_stack([grid[i] for i in np.unravel_index(flat, pi.shape)])


# --------------------------------------------------------------------------- #
# Gaussian copula with polychoric correlations                                 #
# --------------------------------------------------------------------------- #
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
BOUND = 8.0          # stands in for an infinite threshold
RHO_MAX = 0.97


def bvn_cdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal CDF, P(X <= h, Y <= k), arrays broadcast together.

    Uses the single-integral form in the arcsine of the correlation, with 48-point
    Gauss-Legendre quadrature.
    """
    h, k, rho = np.broadcast_arrays(h, k, rho)
    half = np.arcsin(rho) / 2.0
    t = half[..., None] * (_GL_X + 1.0)                      # nodes on [0, asin(rho)]
    s = np.sin(t)
    c2 = np.cos(t) ** 2
    hh = h[..., None]
    kk = k[..., None]
    integrand = np.exp(-(hh * hh + kk * kk - 2.0 * hh * kk * s) / (2.0 * c2))
    return ndtr(h) * ndtr(k) + (integrand @ _GL_W) * half / (2.0 * np.pi)


def thresholds(x: np.ndarray, k: int) -> np.ndarray:
    """Normal thresholds of one item from its answer proportions, with the two ends."""
    cum = np.cumsum(np.bincount(x, minlength=k)) / len(x)
    inner = np.clip(ndtri(np.clip(cum[:-1], 1e-300, 1.0)), -BOUND, BOUND)
    inner = np.where(cum[:-1] >= 1.0, BOUND, inner)
    return np.concatenate([[-BOUND], inner, [BOUND]])


class OrdinalCopula:
    """Thresholds per group, one latent correlation matrix shared by the groups."""

    def fit(self, x: np.ndarray, groups: np.ndarray, n_categories: list[int]) -> "OrdinalCopula":
        n_items = x.shape[1]
        kmax = max(n_categories)
        self.labels_ = np.unique(groups)
        self.n_categories_ = n_categories
        # thresholds padded to kmax categories; a category an item lacks has width zero
        tau = np.full((len(self.labels_), n_items, kmax + 1), BOUND)
        counts = np.zeros((len(self.labels_), n_items, n_items, kmax, kmax))
        for g, label in enumerate(self.labels_):
            xg = x[groups == label]
            for j, k in enumerate(n_categories):
                tau[g, j, :k + 1] = thresholds(xg[:, j], k)
            onehot = np.zeros((len(xg), n_items, kmax))
            onehot[np.arange(len(xg))[:, None], np.arange(n_items)[None, :], xg] = 1.0
            counts[g] = np.einsum("nja,nlb->jlab", onehot, onehot)
        self.tau_ = tau

        ja, la = np.triu_indices(n_items, 1)
        h = tau[:, ja][:, :, :, None]                          # groups x pairs x (k+1) x 1
        k_ = tau[:, la][:, :, None, :]
        cnt = counts[:, ja, la]                                # groups x pairs x k x k

        def loglik(rho: np.ndarray) -> np.ndarray:
            cdf = bvn_cdf(h, k_, rho[None, :, None, None])
            cell = cdf[:, :, 1:, 1:] - cdf[:, :, :-1, 1:] - cdf[:, :, 1:, :-1] + cdf[:, :, :-1, :-1]
            term = np.where(cnt > 0, cnt * np.log(np.clip(cell, 1e-300, None)), 0.0)
            return term.sum(axis=(0, 2, 3))

        # golden-section search for every pair at once
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo = np.full(len(ja), -RHO_MAX)
        hi = np.full(len(ja), RHO_MAX)
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc, fd = loglik(c), loglik(d)
        for _ in range(45):
            left = fc > fd
            hi = np.where(left, d, hi)
            lo = np.where(left, lo, c)
            new_c = hi - gr * (hi - lo)
            new_d = lo + gr * (hi - lo)
            f_new_c, f_new_d = loglik(new_c), loglik(new_d)
            c, d, fc, fd = new_c, new_d, f_new_c, f_new_d
        rho = (lo + hi) / 2.0

        raw = np.eye(n_items)
        raw[ja, la] = rho
        raw[la, ja] = rho
        self.rho_raw_ = raw
        self.at_bound_ = float((np.abs(rho) > RHO_MAX - 1e-3).mean())
        values, vectors = np.linalg.eigh(raw)
        self.min_eigenvalue_ = float(values.min())
        if values.min() < 1e-6:                                # nearest usable correlation matrix
            fixed = (vectors * np.clip(values, 1e-6, None)) @ vectors.T
            scale = np.sqrt(np.diag(fixed))
            raw = fixed / np.outer(scale, scale)
        self.corr_ = raw
        self.chol_ = np.linalg.cholesky(raw)
        return self

    def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((len(groups), len(self.chol_))) @ self.chol_.T
        index = {label: g for g, label in enumerate(self.labels_)}
        out = np.empty(z.shape, dtype=int)
        for i, label in enumerate(groups):
            tau = self.tau_[index[label]]
            for j, k in enumerate(self.n_categories_):
                out[i, j] = min(int(np.searchsorted(tau[j, 1:k], z[i, j])), k - 1)
        return out


# --------------------------------------------------------------------------- #
# proportional-odds regression                                                 #
# --------------------------------------------------------------------------- #
def _po_unpack(par: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    beta = par[:p]
    alpha = par[p] + np.concatenate([[0.0], np.cumsum(np.exp(par[p + 1:]))])
    return beta, alpha


def po_probs(beta: np.ndarray, alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Category probabilities under P(Y <= k | x) = logistic(alpha_k - x beta)."""
    cum = expit(alpha[None, :] - (x @ beta)[:, None])
    upper = np.hstack([cum, np.ones((len(x), 1))])
    lower = np.hstack([np.zeros((len(x), 1)), cum])
    return upper - lower


def po_objective(par: np.ndarray, x: np.ndarray, y: np.ndarray, k: int, lam: float):
    """Penalised negative log-likelihood and its gradient."""
    n, p = x.shape
    beta, alpha = _po_unpack(par, p)
    eta = x @ beta
    cum = expit(alpha[None, :] - eta[:, None])                       # n x (k-1)
    upper = np.hstack([cum, np.ones((n, 1))])
    lower = np.hstack([np.zeros((n, 1)), cum])
    rows = np.arange(n)
    prob = np.clip(upper[rows, y] - lower[rows, y], EPS, None)
    value = -np.log(prob).sum() + 0.5 * lam * beta @ beta

    dens = cum * (1.0 - cum)                                         # d cum / d alpha
    dens_upper = np.hstack([dens, np.zeros((n, 1))])
    dens_lower = np.hstack([np.zeros((n, 1)), dens])
    # d log prob / d eta
    g_eta = -(dens_upper[rows, y] - dens_lower[rows, y]) / prob
    g_beta = -(x.T @ g_eta) + lam * beta
    # d log prob / d alpha_m
    g_alpha = np.zeros(k - 1)
    up = y < k - 1
    np.add.at(g_alpha, y[up], dens_upper[rows[up], y[up]] / prob[up])
    low = y > 0
    np.add.at(g_alpha, y[low] - 1, -dens_lower[rows[low], y[low]] / prob[low])
    g_alpha = -g_alpha
    # chain rule to the unconstrained threshold parameters
    g_par = np.empty(k - 1)
    g_par[0] = g_alpha.sum()
    steps = np.exp(par[p + 1:])
    for m in range(1, k - 1):
        g_par[m] = g_alpha[m:].sum() * steps[m - 1]
    return value, np.concatenate([g_beta, g_par])


def po_fit(x: np.ndarray, y: np.ndarray, k: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    p = x.shape[1]
    cum = np.clip(np.cumsum(np.bincount(y, minlength=k))[:-1] / len(y), 0.005, 0.995)
    alpha0 = np.log(cum / (1 - cum))
    alpha0 = np.maximum.accumulate(alpha0 + 1e-3 * np.arange(k - 1))
    start = np.concatenate([np.zeros(p), [alpha0[0]], np.log(np.clip(np.diff(alpha0), 1e-3, None))])
    res = minimize(po_objective, start, args=(x, y, k, lam), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return _po_unpack(res.x, p)


class OrdinalRegression:
    """One proportional-odds model per item, ridge penalty chosen by cross-validation."""

    grid = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, folds: int = 5) -> "OrdinalRegression":
        """``y`` holds 0-based category indices; an item is modelled on the categories it uses."""
        n, n_items = y.shape
        self.used_ = [np.unique(y[:, j]) for j in range(n_items)]
        dense = [np.searchsorted(self.used_[j], y[:, j]) for j in range(n_items)]
        fold = rng.permutation(n) % folds
        score = np.zeros(len(self.grid))
        for f in range(folds):
            tr, te = fold != f, fold == f
            for j in range(n_items):
                seen = np.unique(dense[j][tr])
                if len(seen) < 2:
                    continue
                y_tr = np.searchsorted(seen, dense[j][tr])
                for g, lam in enumerate(self.grid):
                    beta, alpha = po_fit(x[tr], y_tr, len(seen), lam)
                    prob = np.zeros((int(te.sum()), len(self.used_[j])))
                    prob[:, seen] = po_probs(beta, alpha, x[te])
                    score[g] += np.log(np.clip(prob[np.arange(int(te.sum())), dense[j][te]], 1e-6, None)).sum()
        self.cv_loglik_ = dict(zip(self.grid, score))
        self.lam_ = self.grid[int(np.argmax(score))]
        self.models_ = [po_fit(x, dense[j], len(self.used_[j]), self.lam_) for j in range(n_items)]
        return self

    def probs(self, x: np.ndarray) -> list[np.ndarray]:
        return [po_probs(beta, alpha, x) for beta, alpha in self.models_]

    def sample(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([self.used_[j][sample_categories(p, rng)] for j, p in enumerate(self.probs(x))])

    def median(self, x: np.ndarray) -> np.ndarray:
        out = []
        for j, p in enumerate(self.probs(x)):
            idx = (np.cumsum(p, axis=1) < 0.5).sum(axis=1)
            out.append(self.used_[j][np.minimum(idx, p.shape[1] - 1)])
        return np.column_stack(out)
