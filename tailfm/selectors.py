"""Automatic choice of k for the Hill estimator (Phase 2b comparators, stage S1b).

Both rules were transcribed from their sources on 2026-11-02; see
``docs/sources/selectors-recent.md`` for the equations, page numbers and the choices the
sources leave open. Indices: x_(1) ≥ x_(2) ≥ … are the descending order statistics, and
the Hill estimator at k is H_k = (1/k) Σ_{i≤k} log(x_(i)/x_(k+1)), as in
``tail_estimators.hill_path``.

SAMSEE (Schneider, Krajina & Krivobokova 2021, Extremes 24:881–913, Sect. 3, ρ = −1):

    Y_(i,k) = log(x_(i)/x_(k+1)),  M_k = (1/k) Σ Y²_(i,k),  H^V_k = M_k / (2H_k),
    H^GJ_k = 2H^V_k − H_k                                                     (13)
    b̂_{k,K} = H̄_{k,K} − H̄_{1,K},  H̄_{k,K} = (1/(K−k+1)) Σ_{i=k}^{K} H_i      (14)
    AD(K) = (1/K) Σ_{k=1}^{K} (H^V_k + b̂_{k,K} − H_k)²                         (17)
    K* = argmin_K |Σ_{i∈{−2,−1,1,2}} (AD(K+i) − AD(K))/i|                      (18)
    SAMSEE(k) = (H^GJ_{K*})²/k + 4 b̂²_{k,K*},   k̂ = argmin_{1<k<K*} SAMSEE(k)   (19)

KS distance (Danielsson, Ergun, de Haan & de Vries; Bank of Canada SWP 2019-28,
eqs. 8–9; 2016 working paper eqs. 9–10):

    q(j, k) = x_(k) · (k/j)^{H_k}                                           (Weissman)
    k* = argmin_k max_{1≤j≤T} |x_(j+1) − q(j, k)|
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .preprocessing import Array

# Search range of K for SAMSEE's K* (not stated in the source; fixed here before any
# Phase 2b benchmark result): K ∈ [⌈SAMSEE_K_MIN_FRAC·n⌉, n − 3].
SAMSEE_K_MIN_FRAC = 0.1
# KS distance: T = ⌊KS_TAIL·n⌋ (the 15% of the paper's application and the default of
# R package tea::mindist), k ∈ {2, …, T − 1}.
KS_TAIL = 0.15


def _desc(x: ArrayLike, min_n: int) -> Array:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size < min_n or np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError(f"Require a 1-D positive finite sample with n>={min_n}")
    return np.sort(a)[::-1]


def hill_moments(x: ArrayLike) -> tuple[Array, Array]:
    """(H_k, M_k) for k = 1..n−1 (index k−1): first and second moments of the k largest
    log ratios over x_(k+1)."""
    d = _desc(x, 3)
    logs = np.log(d)
    k = np.arange(1, d.size)
    c1 = np.cumsum(logs)[:-1]
    c2 = np.cumsum(logs**2)[:-1]
    t = logs[1:]  # log x_(k+1)
    h = c1 / k - t
    m = c2 / k - 2.0 * t * c1 / k + t**2
    return h, np.maximum(m, 0.0)


def samsee(x: ArrayLike) -> dict:
    """SAMSEE choice of k (ρ = −1 version). Returns k̂, K*, Hill at k̂."""
    d = _desc(x, 40)
    n = d.size
    h, m = hill_moments(d)
    hv = m / (2.0 * h)
    k_lo = max(4, math.ceil(SAMSEE_K_MIN_FRAC * n))
    k_hi = n - 3
    csum = np.concatenate([[0.0], np.cumsum(h)])  # csum[j] = Σ_{i≤j} H_i

    def bias(big_k: int) -> Array:
        ks = np.arange(1, big_k + 1)
        upper = (csum[big_k] - csum[ks - 1]) / (big_k - ks + 1)
        return upper - csum[big_k] / big_k

    cand = np.arange(k_lo - 2, k_hi + 3)
    ad = np.array([np.mean((hv[:kk] + bias(kk) - h[:kk]) ** 2) for kk in cand])
    centre = np.arange(2, cand.size - 2)
    deriv = sum((ad[centre + i] - ad[centre]) / i for i in (-2, -1, 1, 2))
    k_star = int(cand[centre[int(np.argmin(np.abs(deriv)))]])
    gj = 2.0 * hv[k_star - 1] - h[k_star - 1]
    ks = np.arange(2, k_star)
    crit = gj**2 / ks + 4.0 * bias(k_star)[ks - 1] ** 2
    k_hat = int(ks[int(np.argmin(crit))])
    return {"k": k_hat, "K_star": k_star, "xi": float(h[k_hat - 1])}


def ks_distance(x: ArrayLike, tail: float = KS_TAIL) -> dict:
    """KS-distance choice of k. Returns k*, T and Hill at k*."""
    d = _desc(x, 40)
    n = d.size
    big_t = int(math.floor(tail * n))
    if big_t < 4:
        raise ValueError("Tail region too small")
    h, _ = hill_moments(d)
    ks = np.arange(2, big_t)
    j = np.arange(1, big_t + 1)
    q = d[ks - 1, None] * (ks[:, None] / j[None, :]) ** h[ks - 1, None]
    dist = np.abs(d[j][None, :] - q).max(axis=1)
    k_star = int(ks[int(np.argmin(dist))])
    return {"k": k_star, "T": big_t, "xi": float(h[k_star - 1])}
