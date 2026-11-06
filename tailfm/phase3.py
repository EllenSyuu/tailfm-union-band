"""Phase 3: second-order tail posteriors along a k grid and selection-valid credible bands.

Posterior (docs/theory/T1-epd-posterior-bvm.md): relative excesses Y_i = X_(n−i+1)/X_(n−k),
EPD likelihood with τ = ρ/ξ, prior log ξ uniform on [log 0.02, log 3] and δ ~ N(0, 1)
truncated to δ > max(−1, 1/τ); computed on a grid in (log ξ, δ-prior-CDF).

Selection-valid band (Theorem C3 of docs/theory/C3-simultaneous-bands.md, [Conjecture] until
proved): on a fixed grid K = {k_1 < … < k_m}, report at every k_j the central posterior
interval at the calibrated level 1 − α*, where α* makes the bands hold *simultaneously*
with probability 1 − α under the Gaussian limit. Under an exact Pareto tail the joint law
of the log excesses above X_(n−k_m) is that of ξ·(Σ_{l=i}^{k_m} E_l/l) (Rényi), free of
n and of ξ after scaling, so α* is computed by simulation from Pareto samples of size
k_m + 1, using the EPD estimator of Beirlant, Joossens & Segers (2009) (asymptotically
equivalent to the posterior median by the BvM of T1) and its asymptotic standard
deviation ξ(1 − ρ)/|ρ|/√k.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

from .tail_estimators import epd_path

LOG_XI = (math.log(0.02), math.log(3.0))


def epd_log_marginal(
    x: np.ndarray, k: int, rho: float, *, nx: int = 241, nd: int = 81, width: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """(log ξ grid, log ∫ L(ξ, δ; ρ) dΠ(δ)) for the EPD tail likelihood at k (unnormalised;
    the grid depends only on the data, so values for different ρ are comparable)."""
    s = np.sort(np.log(np.asarray(x, dtype=float)))
    ly = s[-k:] - s[-k - 1]
    h = ly.mean()
    lx = np.log(h) + np.linspace(-width, width, nx)
    lx = lx[(lx > LOG_XI[0]) & (lx < LOG_XI[1])]
    xi = np.exp(lx)[:, None]
    tau = rho / xi
    lower = np.maximum(-1.0, 1.0 / tau)
    p = (np.arange(nd) + 0.5) / nd
    delta = -special.ndtri(special.ndtr(-lower) * (1.0 - p)[None, :])
    out = np.zeros(delta.shape)
    for yv in np.array_split(ly, max(1, k // 256)):
        u = np.exp(tau[..., None] * yv[None, None, :])
        d = delta[..., None]
        a = 1.0 + d - d * u
        b = 1.0 + d - d * (1.0 + tau[..., None]) * u
        with np.errstate(invalid="ignore", divide="ignore"):
            out += (-(1.0 / xi[..., None] + 1.0) * np.log(a) + np.log(b)).sum(-1)
    out += -k * np.log(xi) - (1.0 / xi + 1.0) * ly.sum()
    out = np.where(np.isfinite(out), out, -np.inf)
    m = out.max(1, keepdims=True)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        lp = (safe + np.log(np.exp(out - safe).mean(1, keepdims=True)))[:, 0]
    return lx, lp


def _cdf(lp: np.ndarray) -> np.ndarray:
    w = np.exp(lp - lp.max())
    return np.cumsum(w) / w.sum()


def epd_tail_posterior(x: np.ndarray, k: int, rho: float, **kw) -> tuple[np.ndarray, np.ndarray]:
    """(log ξ grid, posterior CDF of log ξ) for the EPD tail posterior at k, ρ fixed."""
    lx, lp = epd_log_marginal(x, k, rho, **kw)
    return lx, _cdf(lp)


RHO_GRID = -np.exp(np.linspace(np.log(0.05), np.log(3.0), 24))


def epd_tail_posterior_hier(
    x: np.ndarray, k: int, rho_grid: np.ndarray = RHO_GRID, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """EPD tail posterior with ρ integrated out: ρ uniform on the log(−ρ) grid
    (log(−ρ) ~ U(log 0.05, log 3)); returns (log ξ grid, posterior CDF of log ξ)."""
    parts = [epd_log_marginal(x, k, float(r), **kw) for r in rho_grid]
    lx = parts[0][0]
    stack = np.vstack([p for _, p in parts])
    m = stack.max(0)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        lp = safe + np.log(np.exp(stack - safe).mean(0))
    return lx, _cdf(lp)


def quantiles(lx: np.ndarray, cdf: np.ndarray, probs) -> np.ndarray:
    """Quantiles of ξ from a posterior on an equispaced log ξ grid with masses at the grid
    points: each mass is spread over its cell [lx − h/2, lx + h/2], so cdf[i] is the CDF at
    the upper cell edge (using lx itself would shift every quantile by half a cell)."""
    h = lx[1] - lx[0]
    edges = np.concatenate([[lx[0] - h / 2], lx + h / 2])
    return np.exp(np.interp(probs, np.concatenate([[0.0], cdf]), edges))


def pareto_tail_posterior(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """First-order posterior (pure Pareto), same prior on log ξ, for comparison."""
    s = np.sort(np.log(np.asarray(x, dtype=float)))
    ly = s[-k:] - s[-k - 1]
    lx = np.log(ly.mean()) + np.linspace(-2.5, 2.5, 801)
    lx = lx[(lx > LOG_XI[0]) & (lx < LOG_XI[1])]
    lp = -k * lx - ly.sum() / np.exp(lx)
    w = np.exp(lp - lp.max())
    return lx, np.cumsum(w) / w.sum()


def calibrate_level(
    ks, rho: float | None, alpha: float = 0.1, sims: int = 4000, seed: int = 6500
) -> tuple[float, float]:
    """(α*, c): pointwise level α* such that the m intervals ξ̂_k ± z_{α*/2}·sd_k hold
    simultaneously with probability 1 − α under exact Pareto tails (ξ = 1 w.l.o.g.).
    ``rho=None`` calibrates the first-order (Hill / Pareto-posterior) band, sd = 1/√k.
    Development/calibration seeds only (≥ 6500)."""
    ks = np.asarray(ks, dtype=int)
    key = 0 if rho is None else int(1000 * -rho)
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(ks), key]))
    sd = (1.0 if rho is None else (1 - rho) / abs(rho)) / np.sqrt(ks)
    stat = np.empty(sims)
    for s in range(sims):
        x = rng.pareto(1.0, size=ks[-1] + 1) + 1.0  # Pareto(ξ = 1)
        if rho is None:
            logs = np.sort(np.log(x))[::-1]
            est = np.array([logs[:k].mean() - logs[k] for k in ks])
        else:
            est = epd_path(x, ks, rho=rho)
        stat[s] = np.max(np.abs(est - 1.0) / sd)
    c = float(np.quantile(stat, 1 - alpha))
    return float(2 * stats.norm.sf(c)), c


def gp_tail_posterior(
    x: np.ndarray, k: int, *, ng: int = 241, ns: int = 121, width: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Pareto posterior for the k excesses over X_(n−k), as in Dombry, Padoan &
    Rizzelli (arXiv:2310.06720): prior flat in γ on (0.02, 3) (their Condition 2 allows any
    continuous γ prior) and π(σ) ∝ 1/σ. Returns (log γ grid, posterior CDF of log γ) so that
    it is directly comparable with the EPD and Pareto posteriors (here γ = ξ > 0); the flat
    prior in γ is converted to log γ by the Jacobian γ."""
    s = np.sort(np.asarray(x, dtype=float))
    u = s[-k - 1]
    e = s[-k:] - u
    h = np.mean(np.log(s[-k:]) - np.log(u))
    lg = np.log(h) + np.linspace(-width, width, ng)
    lg = lg[(lg > LOG_XI[0]) & (lg < LOG_XI[1])]
    g = np.exp(lg)
    # GP scale is about γ·u for Pareto-type tails; grid in log σ around log(h·u).
    ls = np.log(h * u) + np.linspace(-3.0, 3.0, ns)
    sig = np.exp(ls)
    ll = np.empty((g.size, sig.size))
    for j, sj in enumerate(sig):
        z = np.log1p(g[:, None] * e[None, :] / sj)  # (G, k)
        ll[:, j] = -k * np.log(sj) - (1.0 / g + 1.0) * z.sum(1)
    # prior: flat in γ (Jacobian γ for log γ), 1/σ (flat in log σ on the grid)
    ll += lg[:, None]
    m = ll.max()
    with np.errstate(divide="ignore"):
        lp = np.log(np.exp(ll - m).sum(1)) + m
    return lg, _cdf(lp)


def calibrate_level_gp(
    ks, xi: float, alpha: float = 0.1, sims: int = 800, seed: int = 6600
) -> tuple[float, float]:
    """(α*, c) for the simultaneous GP-posterior band. Under an exact Pareto(ξ) tail the
    excesses over any order statistic are exactly GP(ξ, ξu), so the joint law of the GP
    posteriors along the grid depends on ξ only (not on n or u); calibrated at a given ξ
    (in use ξ̂, asymptotically valid). Statistic: max_j |median_j − ξ| / s_j with s_j the
    posterior scale (90% interval width / (2·1.645)), i.e. the same pivot the band uses —
    the asymptotic sd (1 + ξ)/√k (Drees, Ferreira & de Haan 2004, Cor. 2.1) understates the
    posterior spread at small k. Development/calibration seeds only (≥ 6600)."""
    ks = np.asarray(ks, dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(ks), int(1000 * xi)]))
    stat = np.empty(sims)
    for s in range(sims):
        x = rng.pareto(1.0 / xi, size=ks[-1] + 1) + 1.0  # Pareto with tail index ξ
        z = []
        for k in ks:
            lo, med, hi = quantiles(*gp_tail_posterior(x, int(k), ng=81, ns=41), [0.05, 0.5, 0.95])
            z.append(abs(med - xi) / ((hi - lo) / (2 * 1.6448536269514722)))
        stat[s] = max(z)
    c = float(np.quantile(stat, 1 - alpha))
    return float(2 * stats.norm.sf(c)), c
