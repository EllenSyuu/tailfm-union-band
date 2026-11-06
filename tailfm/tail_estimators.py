"""Classical univariate tail-index estimators for Phase 2, each evaluated on a grid of k.

Every estimator here returns one estimate per threshold k so that an evaluation can
choose k post hoc. References for formulas:

- Hill (1975).
- Reduced-bias Hill, MVRB: Caeiro, Gomes & Pestana (2005), with ρ̂ from Fraga Alves,
  Gomes & de Haan (2003) and β̂ from Gomes & Martins (2002), both at k₁ = ⌊n^0.999⌋.
- EPD: Beirlant, Joossens & Segers (2009), δ̂ and γ̂ from their linearised score
  equations, with τ̂ = ρ̂/H(k).
- Bayes-Hill: posterior median of ξ given the Hill statistic under a uniform prior
  (exact for Pareto tails; Phase 1).
- EPD-Bayes: posterior median of ξ under the extended Pareto likelihood of the k
  relative excesses, uniform priors on ξ, ρ and δ, on a grid (prior-aware control).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tails import TRAIN_RHO, TRAIN_XI

Array = NDArray[np.float64]


def _sorted_logs(x: ArrayLike) -> Array:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size < 16 or np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError("Require a 1-D positive finite sample with n>=16")
    return np.log(np.sort(a))


def _check_ks(ks: ArrayLike, n: int) -> NDArray[np.int64]:
    k = np.asarray(ks, dtype=np.int64)
    if k.ndim != 1 or k.size == 0 or np.any(k < 2) or np.any(k >= n):
        raise ValueError("Require thresholds 2<=k<n")
    return k


def hill_path(x: ArrayLike, ks: ArrayLike) -> Array:
    """Hill estimates H(k) = mean of the k largest log(X_(n−i+1)/X_(n−k)), one per k."""
    logs = _sorted_logs(x)
    n = logs.size
    k = _check_ks(ks, n)
    top = logs[::-1]  # descending
    csum = np.cumsum(top)
    return csum[k - 1] / k - top[k]


def second_order_estimates(x: ArrayLike, k1: int | None = None) -> tuple[float, float]:
    """(ρ̂, β̂) at the high level k₁ (default ⌊n^0.999⌋).

    ρ̂ uses τ=0 when that estimate is ≥ −1 and τ=1 otherwise (the usual practical rule
    for |ρ|≤1 versus |ρ|>1). β̂ uses the scaled log-spacings U_i = i·(log X_(n−i+1) −
    log X_(n−i)). Both are consistent under the Hall second-order condition.
    """
    logs = _sorted_logs(x)
    n = logs.size
    k1 = int(math.floor(n**0.999)) if k1 is None else int(k1)
    if not 16 <= k1 < n:
        raise ValueError("Require 16<=k1<n")
    top = logs[::-1]
    v = top[:k1] - top[k1]
    m1, m2, m3 = v.mean(), (v**2).mean(), (v**3).mean()

    def rho_hat(tau: int) -> float:
        if tau == 0:
            a, b, c = math.log(m1), 0.5 * math.log(m2 / 2), math.log(m3 / 6) / 3
        else:
            a, b, c = m1, math.sqrt(m2 / 2), (m3 / 6) ** (1 / 3)
        t = (a - b) / (b - c)
        return -abs(3 * (t - 1) / (t - 3))

    rho = rho_hat(0)
    if rho < -1:
        rho = rho_hat(1)
    i = np.arange(1, k1 + 1, dtype=np.float64)
    u = i * (top[:k1] - top[1 : k1 + 1])
    w = (i / k1) ** (-rho)
    d1, d2 = w.mean(), (w * u).mean()
    numerator = d1 * u.mean() - d2
    denominator = d1 * d2 - ((i / k1) ** (-2 * rho) * u).mean()
    beta = (k1 / n) ** rho * numerator / denominator
    return float(rho), float(beta)


def mvrb_path(x: ArrayLike, ks: ArrayLike, k1: int | None = None) -> Array:
    """Reduced-bias Hill H(k)·(1 − β̂/(1−ρ̂)·(n/k)^ρ̂) with (ρ̂, β̂) at k₁."""
    n = np.asarray(x).size
    k = _check_ks(ks, n)
    rho, beta = second_order_estimates(x, k1)
    return hill_path(x, k) * (1 - beta / (1 - rho) * (n / k) ** rho)


def epd_path(x: ArrayLike, ks: ArrayLike, rho: float | None = None) -> Array:
    """EPD estimates γ̂(k) = H − δ̂ρ̂/(1−ρ̂), δ̂ from E_k(ρ̂/H) (Beirlant et al., 2009)."""
    logs = _sorted_logs(x)
    n = logs.size
    k = _check_ks(ks, n)
    r = second_order_estimates(x)[0] if rho is None else float(rho)
    if not r < 0:
        raise ValueError("Require rho<0")
    top = logs[::-1]
    out = np.empty(k.size)
    for j, kk in enumerate(k):
        v = top[:kk] - top[kk]
        h = v.mean()
        e = np.exp(r / h * v).mean()
        delta = h * (1 - 2 * r) * (1 - r) ** 3 * r**-4 * (e - 1 / (1 - r))
        out[j] = h - delta * r / (1 - r)
    return out


def _posterior_median(log_post: Array, grid: Array) -> float:
    w = np.exp(log_post - log_post.max())
    cdf = np.cumsum(w) / w.sum()
    return float(grid[min(int(np.searchsorted(cdf, 0.5)), grid.size - 1)])


def bayes_hill_path(
    x: ArrayLike, ks: ArrayLike, prior: tuple[float, float] = TRAIN_XI, grid_size: int = 1001
) -> Array:
    """Posterior median of ξ given k·H(k) ~ Gamma(k, ξ), uniform prior; one per k."""
    n = np.asarray(x).size
    k = _check_ks(ks, n)
    grid = np.linspace(*prior, grid_size)
    totals = hill_path(x, k) * k
    return np.array(
        [
            _posterior_median(-kk * np.log(grid) - t / grid, grid)
            for kk, t in zip(k, totals, strict=True)
        ]
    )


def bayes_burr(
    z: ArrayLike,
    n: int,
    *,
    xi_prior: tuple[float, float] = TRAIN_XI,
    rho_prior: tuple[float, float] = TRAIN_RHO,
    grid: tuple[int, int, int] = (101, 13, 100),
) -> float:
    """Exact Bayes estimator of ξ under the Burr training model, from log ratios alone.

    ``z`` holds the K largest log(X_(n−i+1)/X_(n−K)) of a sample of size n — exactly the
    network's input. Log ratios are scale invariant, so their density under Burr(ξ, ρ)
    with any scale is obtained by integrating over the survival probability v of the
    threshold X_(n−K):

        p(z | ξ, ρ) ∝ ∫₀¹ (1−v)^(n−K−1) ∏ᵢ f(Q(v)·e^{zᵢ}) · Q(v)·e^{zᵢ} dv,

    with Burr density f(x) = ξ⁻¹ x^(a−1) (1+x^a)^(1/ρ−1), a = −ρ/ξ, and quantile
    Q(v) = (v^ρ − 1)^(−ξ/ρ). Uniform priors on ξ and ρ (the training box); ρ is
    marginalised. Returns the posterior median, which minimises expected absolute
    error: in the training model no estimator using these log ratios can do better
    on average, the network included.
    """
    zz = np.asarray(z, dtype=np.float64)
    if zz.ndim != 1 or zz.size < 2 or np.any(zz < 0) or zz.size >= n:
        raise ValueError("Require nonnegative log ratios with 2<=K<n")
    kk = zz.size
    m = n - kk - 1
    xs = np.linspace(*xi_prior, grid[0])[:, None, None]
    rs = np.linspace(*rho_prior, grid[1])[None, :, None]
    # Survival level of the threshold; (1-v)^m concentrates near v ≈ K/n.
    centre = (kk + 1) / (n + 1)
    lv = np.linspace(np.log(centre / 8), np.log(min(0.95, centre * 4)), grid[2])
    v = np.exp(lv)[None, None, :]
    r = rs[0, :, 0][:, None, None]  # (R, 1, 1)
    vv = v[0, 0, :][None, :, None]  # (1, V, 1)
    base = m * np.log1p(-v[0, 0, :]) + lv  # (V,)
    log_post = np.empty((grid[0], grid[1]))
    for i in range(grid[0]):
        xi_ = xs[i, 0, 0]
        a = -r / xi_
        lx = (-xi_ / r) * np.log(vv**r - 1) + zz[None, None, :]  # (R, V, K)
        log_f = -np.log(xi_) + (a - 1) * lx + (1 / r - 1) * np.logaddexp(0, a * lx)
        integrand = base[None, :] + (log_f + lx).sum(axis=2)  # (R, V)
        peak = integrand.max(axis=1, keepdims=True)
        log_post[i] = (peak + np.log(np.exp(integrand - peak).sum(axis=1, keepdims=True)))[:, 0]
    peak = log_post.max()
    with np.errstate(divide="ignore"):
        marginal = np.log(np.exp(log_post - peak).sum(axis=1)) + peak
    return _posterior_median(marginal, xs[:, 0, 0])


def epd_bayes_path(
    x: ArrayLike,
    ks: ArrayLike,
    *,
    xi_prior: tuple[float, float] = TRAIN_XI,
    rho_prior: tuple[float, float] = TRAIN_RHO,
    delta_prior: tuple[float, float] = (-0.95, 3.0),
    grid: tuple[int, int, int] = (81, 9, 21),
) -> Array:
    """Posterior median of ξ under the EPD likelihood of the k relative excesses.

    EPD density (Beirlant, Joossens & Segers 2009) for y>1 with τ=ρ/ξ:
    g(y) = ξ⁻¹ y^(−1/ξ−1) {1+δ(1−y^τ)}^(−1/ξ−1) [1+δ{1−(1+τ)y^τ}].
    Uniform priors on ξ, ρ and δ, integrated on a grid; parameter combinations
    that make the density invalid on the observed excesses get zero prior mass.
    ρ and δ are nuisance parameters marginalised out. The ξ and ρ priors default to
    the network's training box, so this control shares the network's prior knowledge.
    """
    logs = _sorted_logs(x)
    n = logs.size
    k = _check_ks(ks, n)
    xs = np.linspace(*xi_prior, grid[0])[:, None, None, None]
    rs = np.linspace(*rho_prior, grid[1])[None, :, None, None]
    ds = np.linspace(*delta_prior, grid[2])[None, None, :, None]
    top = logs[::-1]
    out = np.empty(k.size)
    for j, kk in enumerate(k):
        logy = (top[:kk] - top[kk])[None, None, None, :]
        tau = rs / xs
        ytau = np.exp(tau * logy)
        a = 1 + ds * (1 - ytau)
        b = 1 + ds * (1 - (1 + tau) * ytau)
        valid = np.all((a > 0) & (b > 0), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = (
                -kk * np.log(xs[..., 0])
                - (1 / xs[..., 0] + 1) * logy.sum(-1)
                - (1 / xs[..., 0] + 1) * np.log(np.where(a > 0, a, 1)).sum(-1)
                + np.log(np.where(b > 0, b, 1)).sum(-1)
            )
        ll = np.where(valid, ll, -np.inf)
        flat = ll.reshape(grid[0], -1)
        peak = flat.max()
        with np.errstate(divide="ignore"):
            marginal = np.log(np.exp(flat - peak).sum(axis=1)) + peak
        out[j] = _posterior_median(marginal, xs[:, 0, 0, 0])
    return out


def epd_loglik(logy: ArrayLike, xi: float, delta: float, tau: float) -> float:
    """EPD log-likelihood of relative excesses, eq. (8) of Maribe, Verster & Beirlant.

    l = −k log ξ − (1/ξ + 1) Σ [log y + log(1 + δ(1 − y^τ))] + Σ log(1 + δ(1 − (1+τ)y^τ)),
    the density of F̄(y) = {y(1 + δ − δy^τ)}^(−1/ξ), y > 1 (their eq. 6). Returns −inf
    where either logarithm's argument is not positive (density invalid on the data).
    Verified against arXiv:1606.05687 (version dated 2018-09-25) on 2026-11-01.
    """
    ly = np.asarray(logy, dtype=np.float64)
    if not xi > 0:
        return -np.inf
    ytau = np.exp(tau * ly)
    a = 1.0 + delta * (1.0 - ytau)
    b = 1.0 + delta * (1.0 - (1.0 + tau) * ytau)
    if np.any(a <= 0) or np.any(b <= 0):
        return -np.inf
    k = ly.size
    return float(-k * math.log(xi) - (1 / xi + 1) * (ly.sum() + np.log(a).sum()) + np.log(b).sum())


def bayes_epd_path(
    x: ArrayLike,
    ks: ArrayLike,
    *,
    rho: float | None = None,
    sigma2: ArrayLike | None = None,
    grid: tuple[int, int] = (60, 61),
) -> Array:
    """Bayesian EPD estimator of Maribe, Verster & Beirlant (arXiv:1606.05687): one ξ̂ per k.

    Prior art for Phase 2b. Formulas verified against the paper (2018-09-25 version):

    - likelihood of the k relative excesses: :func:`epd_loglik` (their eq. 8), with τ
      plugged in as τ̂ = ρ̂/H_{k,n} and ρ̂ = min(−0.5, ρ̂_Fraga) (Section 3);
    - MDI prior on ξ: π(ξ) ∝ e^(−ξ)/ξ (eq. 10);
    - prior on δ: N(0, σ²_{k,n}) truncated to δ > max(−1, 1/τ) (eq. 11), with
      σ²_{k,n} = (k/n)^(−2ρ̂) (eq. 23);
    - point estimate: posterior mode of ξ.

    Deviations from the paper, both computational: the paper takes the mode of OpenBUGS
    MCMC draws and smooths the k-path with a moving average of width 5; here the joint
    posterior mode of (ξ, δ) is found by a grid search refined with Nelder–Mead, and no
    smoothing is applied (so the estimator stays a function of one k). The mode is the
    global maximiser and may lie next to the truncation point of the δ prior. ρ̂_Fraga is the
    Fraga Alves–Gomes–de Haan estimator of :func:`second_order_estimates` at
    k₁ = ⌊n^0.999⌋; the paper does not state k₁ (unverified detail). ``rho`` and
    ``sigma2`` (one value or one per k) override ρ̂ and σ² for tests.
    """
    from scipy.optimize import minimize

    logs = _sorted_logs(x)
    n = logs.size
    k = _check_ks(ks, n)
    if rho is None:
        rho = min(-0.5, second_order_estimates(x)[0])
    if not rho < 0:
        raise ValueError("Require rho<0")
    s2 = (k / n) ** (-2 * rho) if sigma2 is None else np.broadcast_to(sigma2, k.shape)
    top = logs[::-1]
    hills = hill_path(x, k)
    out = np.empty(k.size)
    for j, kk in enumerate(k):
        ly = top[:kk] - top[kk]
        tau = rho / hills[j]
        lower = max(-1.0, 1.0 / tau)
        var = float(s2[j])

        def neg_post(p, ly=ly, tau=tau, lower=lower, var=var):
            xi_, d = p
            if not xi_ > 0 or not d > lower:
                return np.inf
            ll = epd_loglik(ly, xi_, d, tau)
            return -(ll - xi_ - math.log(xi_) - 0.5 * d * d / var)

        sd = math.sqrt(var)
        xis = hills[j] * np.exp(np.linspace(-1.5, 1.5, grid[0]))
        ds = np.linspace(max(lower + 1e-6, -4 * sd), 4 * sd, grid[1])
        if lower > -4 * sd:
            # The posterior can peak in a narrow ridge at the truncation point δ → lower
            # (found on Burr development data); resolve it with points next to it.
            ds = np.union1d(ds, lower + np.geomspace(1e-7, 0.1, 15))
        vals = np.array([[neg_post((a, d)) for d in ds] for a in xis])
        i0, j0 = np.unravel_index(np.argmin(vals), vals.shape)
        res = minimize(
            neg_post,
            x0=[xis[i0], ds[j0]],
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000},
        )
        out[j] = res.x[0] if res.fun <= vals[i0, j0] else xis[i0]
    return out
