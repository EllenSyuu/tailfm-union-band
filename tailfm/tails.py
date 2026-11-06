"""Univariate heavy-tailed families with known extreme-value index and second-order parameter.

Phase 2 studies estimation of ξ when tails are not exactly Pareto. Every family here
has a regularly varying upper tail with index ξ>0 and a known second-order parameter
ρ≤0, so the bias of threshold-based estimators is controlled and known. All families
except log-gamma are in the Hall class with ρ<0; log-gamma has a slowly varying factor
log x (ρ=0) and is included in Phase 2b for robustness only.
All draws use a caller-owned ``numpy.random.Generator`` and return float64 samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from .preprocessing import Array

FAMILIES_2 = ("burr", "frechet", "abs_t", "gpd", "pareto")
# Phase 2b order (docs/phase-2b.md). Phase 2a seed namespaces index FAMILIES_2, which
# must therefore never change; Phase 2b namespaces index FAMILIES_2B.
FAMILIES_2B = (*FAMILIES_2, "pareto_mix", "log_gamma")
# Shape of the log-gamma family: log X ~ Gamma(LOG_GAMMA_SHAPE, scale ξ). Shape 2 gives the
# closed form F̄(x) = (1 + log(x)/ξ)·x^(−1/ξ) on x ≥ 1 (fixed before any Phase 2b result).
LOG_GAMMA_SHAPE = 2.0
TRAIN_XI = (0.05, 1.0)
TRAIN_RHO = (-2.0, -0.25)


@dataclass(frozen=True)
class TailSample:
    """One iid sample with its generating tail parameters."""

    x: Array
    family: str
    xi: float
    rho: float


def second_order(family: str, xi: float, rho: float | None = None) -> float:
    """Second-order parameter ρ of a family (−inf for exact Pareto, which has no bias)."""
    if family == "burr":
        if rho is None or not rho < 0:
            raise ValueError("Burr requires rho<0")
        return float(rho)
    return {
        "frechet": -1.0,
        "abs_t": -2.0 * xi,
        "gpd": -xi,
        "pareto": -np.inf,
        "pareto_mix": -1.0,
        "log_gamma": 0.0,
    }[family]


def survival(family: str, x: ArrayLike, xi: float, rho: float | None = None) -> Array:
    """Exact survival function P(X > x) of the samples produced by :func:`draw`.

    pareto_mix: 0.5·x^(−1/ξ) + 0.5·x^(−2/ξ) on x ≥ 1 = 0.5·x^(−1/ξ)(1 + x^(−1/ξ)),
                a Hall-class tail with ρ = −1 and a positive second-order term (δ > 0).
    log_gamma:  (1 + log(x)/ξ)·x^(−1/ξ) on x ≥ 1, i.e. log X ~ Gamma(2, scale ξ); the
                factor log x is slowly varying, so ρ = 0 (outside the Hall class).
    """
    if family not in FAMILIES_2B:
        raise ValueError(f"Unknown family {family!r}")
    r = second_order(family, xi, rho)
    t = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if family == "burr":
            out = (1.0 + t ** (-r / xi)) ** (1.0 / r)
        elif family == "frechet":
            out = -np.expm1(-(t ** (-1.0 / xi)))
        elif family == "abs_t":
            out = 2.0 * stats.t.sf(t, 1.0 / xi)
        elif family == "gpd":
            out = (1.0 + xi * (t - 1.0)) ** (-1.0 / xi)
        elif family == "pareto":
            out = t ** (-1.0 / xi)
        elif family == "pareto_mix":
            out = 0.5 * t ** (-1.0 / xi) + 0.5 * t ** (-2.0 / xi)
        else:
            out = special.gammaincc(LOG_GAMMA_SHAPE, np.log(t) / xi)
    lower = 0.0 if family in ("burr", "frechet", "abs_t") else 1.0
    return np.where(t <= lower, 1.0, out)


def draw(family: str, n: int, xi: float, rng: np.random.Generator, rho: float | None = None):
    """Draw n positive observations from ``family`` with extreme-value index ``xi``.

    burr:    survival (1 + x^(−ρ/ξ))^(1/ρ), quantile ((1−U)^ρ − 1)^(−ξ/ρ)
    frechet: survival 1 − exp(−x^(−1/ξ)), quantile (−log U)^(−ξ)
    abs_t:   |T| with T ~ Student-t(ν=1/ξ)
    gpd:     survival (1 + ξx)^(−1/ξ), quantile ((1−U)^(−ξ) − 1)/ξ, shifted by 1 to stay positive
    pareto:  survival x^(−1/ξ) on x≥1
    pareto_mix: equal mixture of Pareto(ξ) and Pareto(ξ/2), survival 0.5x^(−1/ξ) + 0.5x^(−2/ξ)
    log_gamma: exp(Y), Y = ξ·Q⁻¹(2, U) ~ Gamma(2, scale ξ) by inversion of the upper
             regularised incomplete gamma function
    """
    if family not in FAMILIES_2B:
        raise ValueError(f"Unknown family {family!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not np.isfinite(xi) or xi <= 0:
        raise ValueError("Require finite xi>0")
    r = second_order(family, xi, rho)
    survival = rng.uniform(size=n)
    if family == "burr":
        x = (survival**r - 1.0) ** (-xi / r)
    elif family == "frechet":
        x = (-np.log1p(-survival)) ** (-xi)
    elif family == "abs_t":
        x = np.abs(rng.standard_t(1.0 / xi, size=n))
    elif family == "gpd":
        x = 1.0 + np.expm1(-xi * np.log(survival)) / xi
    elif family == "pareto_mix":
        heavy = rng.uniform(size=n) < 0.5
        x = survival ** (-np.where(heavy, xi, 0.5 * xi))
    elif family == "log_gamma":
        x = np.exp(xi * special.gammainccinv(LOG_GAMMA_SHAPE, survival))
    else:
        x = survival ** (-xi)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise FloatingPointError("Sample left the positive finite range; reduce xi or n")
    return TailSample(np.asarray(x, dtype=np.float64), family, float(xi), r)


def draw_from_prior(
    n: int,
    rng: np.random.Generator,
    *,
    family: str = "burr",
    xi_range: tuple[float, float] = TRAIN_XI,
    rho_range: tuple[float, float] = TRAIN_RHO,
) -> TailSample:
    """Draw (ξ, ρ) uniformly from the given box, then a sample; ρ is used only for Burr."""
    xi = float(rng.uniform(*xi_range))
    rho = float(rng.uniform(*rho_range)) if family == "burr" else None
    return draw(family, n, xi, rng, rho)


def top_log_ratios(x: Array, k: int) -> Array:
    """The k largest log(X_(n−i+1)/X_(n−k)), ascending; scale invariant, shape-bearing."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not 1 <= k < a.size or np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError("Require a 1-D positive finite sample and 1<=k<n")
    s = np.sort(a)
    return np.log(s[-k:]) - np.log(s[-k - 1])
