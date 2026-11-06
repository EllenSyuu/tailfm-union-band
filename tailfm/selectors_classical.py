"""Classical automatic choices of k for the Hill estimator (Phase 2b comparators, S1c).

Transcribed on 2026-11-02 from the papers supplied by the owner; equations, page
numbers and the choices the papers leave open are in ``docs/sources/selectors-classical.md``.
Our indices: d_1 ≥ d_2 ≥ … ≥ d_n descending order statistics, H_k = (1/k) Σ_{i≤k}
log(d_i/d_{k+1}) (k log ratios).

Drees & Kaufmann (1998), SPA 75:149–172. Their Hill estimator β̂_{n,k} uses k − 1 log
ratios over X_{n−k+1:n} (p. 150), so β̂_{n,k} = H_{k−1}; this module keeps their index k.

    k̄(r) = min{k ∈ {2,…,n}: max_{2≤i≤k} i^{1/2}|β̂_{n,i} − β̂_{n,k}| > r}            (4)
    ρ̂⁽¹⁾_λ(r) = log[ max_{2≤i≤[λk̄]} i^{1/2}|β̂_{n,i} − β̂_{n,[λk̄]}|
                     / max_{2≤i≤k̄} i^{1/2}|β̂_{n,i} − β̂_{n,k̄}| ] / log λ − 1/2,  k̄ = k̄(r)
    k̂ = [(2ρ̂+1)^{−1/ρ̂} (2β̃²ρ̂)^{1/(2ρ̂+1)} (k̄(r^ξ)/k̄(r)^ξ)^{1/(1−ξ)}]           (Thm 1)
    Section 3: r_n = 2.5 β̃ n^{1/4}, ξ = 0.7, β̃ = Hill on the upper ⌊2√n⌋ order
    statistics, r_n → 0.9 r_n until k̄ is defined; ρ either fixed ρ₀ = 1 or
    ρ̂ = ρ̂⁽¹⁾_{0.6}(r_n^ξ) (ρ̂ := 1 when undefined). Their ρ > 0 is minus ours.

Guillou & Hall (2001), JRSS-B 63:293–305, Sect. 2.2–3 (upper tail):

    Y_i = i log(d_i/d_{i+1}),  H_k = k⁻¹ Σ_{i≤k} Y_i                                (2.1)
    w_i = sgn(k − 2i + 1)|k − 2i + 1|^p,  U_k = Σ_{i≤k} w_i Y_i,
    T_k = {(2p+1)/k^{2p+1}}^{1/2} H_k⁻¹ U_k                                         (2.3)
    Q_k = {(2l+1)⁻¹ Σ_{j=−l}^{l} T²_{k+j}}^{1/2},  l = ⌊k/2⌋                       (2.6)
    k̂ = least k with Q_k > c_crit (see guillou_hall);  p = 1, c_crit = 1.25 (Sect. 3)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .preprocessing import Array
from .selectors import _desc, hill_moments

DK_XI = 0.7
DK_R_FACTOR = 2.5
DK_LAMBDA = 0.6
GH_P = 1.0
GH_CRIT = 1.25


def _dk_beta(d: Array) -> Array:
    """β̂_{n,k} for k = 0..n (entries 0 and 1 unused): β̂_{n,k} = H_{k−1}."""
    h, _ = hill_moments(d)
    out = np.full(d.size + 1, np.nan)
    out[2:] = h
    return out


def _dk_stat(beta: Array) -> Array:
    """S(k) = max_{2≤i≤k} i^{1/2}|β̂_{n,i} − β̂_{n,k}| for k = 2..n (index k), in blocks."""
    n = beta.size - 1
    ks = np.arange(2, n + 1)
    s = np.full(n + 1, np.nan)
    root = np.sqrt(np.arange(n + 1.0))
    for lo in range(2, n + 1, 256):
        kk = ks[(ks >= lo) & (ks < lo + 256)]
        i = np.arange(2, kk[-1] + 1)
        dev = root[i][None, :] * np.abs(beta[i][None, :] - beta[kk][:, None])
        dev = np.where(i[None, :] <= kk[:, None], dev, 0.0)
        s[kk] = dev.max(axis=1)
    return s


def _k_bar(stat: Array, r: float) -> int | None:
    over = np.nonzero(stat[2:] > r)[0]
    return int(over[0] + 2) if over.size else None


def _k_bar_shrunk(stat: Array, r: float) -> tuple[int, float]:
    """k̄(r), replacing r by 0.9r until it is defined (Sect. 3)."""
    for _ in range(10_000):
        k = _k_bar(stat, r)
        if k is not None:
            return k, r
        r *= 0.9
    raise FloatingPointError("k̄ undefined")


def drees_kaufmann(x: ArrayLike, rho: float | None = None) -> dict:
    """Drees–Kaufmann adaptive Hill estimator. ``rho=None`` estimates ρ by ρ̂⁽¹⁾_{0.6}(r^ξ);
    a number fixes ρ₀ (their sign, ρ₀ > 0). Returns k in OUR index (number of log ratios),
    the Hill estimate, ρ used, and whether the procedure failed (k̂ < 2) and fell back to
    the initial estimator β̃."""
    d = _desc(x, 40)
    n = d.size
    beta = _dk_beta(d)
    stat = _dk_stat(beta)
    k0 = int(math.floor(2.0 * math.sqrt(n)))
    b0 = float(beta[k0])
    kb_r, r_used = _k_bar_shrunk(stat, DK_R_FACTOR * b0 * n**0.25)
    kb_rx, _ = _k_bar_shrunk(stat, r_used**DK_XI)
    rho_hat = rho
    if rho_hat is None:
        lam_k = int(math.floor(DK_LAMBDA * kb_rx))
        rho_hat = math.nan
        if lam_k >= 2 and stat[kb_rx] > 0:
            num = stat[lam_k]
            with np.errstate(divide="ignore"):
                rho_hat = math.log(num / stat[kb_rx]) / math.log(DK_LAMBDA) - 0.5
        if not (np.isfinite(rho_hat) and rho_hat > 0):
            rho_hat = 1.0
    rh = float(rho_hat)
    ratio = (kb_rx / kb_r**DK_XI) ** (1.0 / (1.0 - DK_XI))
    k_raw = (2 * rh + 1) ** (-1 / rh) * (2 * b0**2 * rh) ** (1 / (2 * rh + 1)) * ratio
    k_dk = int(math.floor(k_raw)) if np.isfinite(k_raw) else 0
    failed = k_dk < 2
    if failed:
        k_dk = k0
    k_dk = min(k_dk, n)
    return {"k": k_dk - 1, "xi": float(beta[k_dk]), "rho": rh, "failed": bool(failed)}


def guillou_hall_stats(x: ArrayLike, p: float = GH_P) -> tuple[Array, Array]:
    """(T_k, Q_k) for k = 1..n−1 (index k; entry 0 unused, Q NaN where undefined)."""
    d = _desc(x, 3)
    n = d.size
    i = np.arange(1, n)
    y = i * np.log(d[:-1] / d[1:])  # Y_i, i = 1..n−1
    t = np.full(n, np.nan)
    for k in range(1, n):
        w = np.sign(k - 2 * i[:k] + 1) * np.abs(k - 2 * i[:k] + 1) ** p
        hk = y[:k].mean()
        t[k] = math.sqrt((2 * p + 1) / k ** (2 * p + 1)) * (w @ y[:k]) / hk
    q = np.full(n, np.nan)
    sq = np.concatenate([[0.0], np.cumsum(np.nan_to_num(t[1:] ** 2))])  # sq[k] = Σ_{m≤k}
    for k in range(1, n):
        lo = k // 2
        if k - lo >= 1 and k + lo <= n - 1:
            q[k] = math.sqrt((sq[k + lo] - sq[k - lo - 1]) / (2 * lo + 1))
    return t, q


def guillou_hall(x: ArrayLike, crit: float = GH_CRIT) -> dict:
    """Guillou–Hall choice of k (Q-version, p = 1, l = ⌊k/2⌋): the smallest k at which Q_k
    exceeds ``crit`` ("the bias is significant when k is so large that Q_k exceeds a
    critical point", p. 296). This reading reproduces their Table 1; the stricter reading
    "Q_m > crit for all m ≥ k" (stated on p. 296 for |T_k|) gives about twice their
    average k̂ (``docs/sources/selectors-classical.md``). If Q never exceeds ``crit``, k̂ is
    the largest k at which Q is defined and ``failed`` is set."""
    d = _desc(x, 40)
    _, q = guillou_hall_stats(d)
    ks = np.nonzero(np.isfinite(q))[0]
    ks = ks[ks >= 2]
    over = ks[q[ks] > crit]
    failed = over.size == 0
    k_hat = int(ks[-1]) if failed else int(over[0])
    h, _ = hill_moments(d)
    return {"k": k_hat, "xi": float(h[k_hat - 1]), "failed": bool(failed)}
