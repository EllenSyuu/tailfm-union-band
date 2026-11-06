"""Quantities for the uniform endpoint bound of the union band (preprint Theorem 3).

Theorem 3: for every ρ in R = [ρ_1, ρ_L] and every λ, Γ_M(ρ, λ) ≥ 1 − φ, where, with
Z_jl = G_jl / s_l,
    φ = P({max_j Z_j1 > c_1} ∪ {min_j Z_jL < −c_L}) ≤ π_1 + π_L,  π_l = P(max_j Z_jl > c_l).
This script evaluates, in the exact Gaussian limit (closed-form covariance), for
(a) the implemented two-sided critical values and (b) the one-sided endpoint variant
(c_1, c_L set to the (1 − α/2)-quantile of max_j Z_jl; interior points unchanged):
    c_l, π_l, the downward analogue, q_l = P(max_j Z_jl > c_l, min_j Z_jl < −c_l), φ,
    and min Γ_M over the grid of (ρ, λ) used in `theory_union_limit.py` (variant (b) only;
    variant (a) is in union_limit.json).
It also reports the finite-k skewness of the endpoint band under exact Pareto tails at
k = 64…1024 (upward vs downward exceedance with the two-sided constant).

Usage: python -I scripts/theory_union_endpoint_bound.py OUTDIR
Seeds (development): Gaussian 6492 (calibration) / 6493 (evaluation); Pareto 6490 / 6491.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from theory_union_limit import beta, g_gauss, g_paths, mesh  # noqa: E402

ALPHA = 0.1
KS = np.array([64, 128, 256, 512, 1024])


def endpoint_stats(zc: np.ndarray, ze: np.ndarray, one_sided: bool) -> tuple[np.ndarray, dict]:
    nm = zc.shape[2]
    c = np.quantile(np.abs(zc).max(1), 1 - ALPHA, axis=0)
    if one_sided:
        cq = np.quantile(zc.max(1), 1 - ALPHA / 2, axis=0)
        ends = np.isin(np.arange(nm), [0, nm - 1])
        c = np.where(ends, cq, c)
    up = ze.max(1) > c
    dn = ze.min(1) < -c
    out = {
        "c": c.tolist(),
        "two_sided_exceed": (up | dn).mean(0).tolist(),
        "pi_up": up.mean(0).tolist(),
        "pi_down": dn.mean(0).tolist(),
        "q": (up & dn).mean(0).tolist(),
        "phi_plus": float((up[:, 0] | dn[:, -1]).mean()),
        "phi_minus": float((up[:, -1] | dn[:, 0]).mean()),
    }
    return c, out


def main() -> None:
    out_dir = Path(sys.argv[1])
    t = KS / KS[-1]
    m6 = np.sort(mesh(6))
    s = (1 - m6) / np.abs(m6)
    n = 1_000_000
    zc = g_gauss(np.random.default_rng(6492), n, KS, m6) / s
    ze = g_gauss(np.random.default_rng(6493), n, KS, m6) / s
    c2, two = endpoint_stats(zc, ze, one_sided=False)
    c1, one = endpoint_stats(zc, ze, one_sided=True)
    one["width_factor_endpoints"] = (c1[[0, -1]] / c2[[0, -1]]).tolist()
    # min of Γ_M for the one-sided variant on the grid of theory_union_limit.py
    g = ze[:50_000] * s
    rhos = -np.exp(np.linspace(math.log(0.5), math.log(2.0), 61))
    lams = np.union1d(np.linspace(-12, 12, 49), [-48, -32, -24, -16, 16, 24, 32, 48])
    worst = (2.0, None)
    for r in rhos:
        b = beta(r, m6)
        for lam in lams:
            sh = (lam * t ** (0.5 - r))[:, None] * b[None, :]
            lo = (g + sh - c1 * s).min(2)
            hi = (g + sh + c1 * s).max(2)
            cov = float(np.mean(np.all((lo <= 0) & (hi >= 0), 1)))
            if cov < worst[0]:
                worst = (cov, [float(r), float(lam)])
    one["Gamma_min"] = worst[0]
    one["Gamma_min_at"] = worst[1]
    one["Gamma_min_paths"] = int(g.shape[0])
    # finite-k skewness under exact Pareto tails at the same k
    npar = 200_000
    pc = g_paths(np.random.default_rng(6490), npar, KS, m6) / s
    pe = g_paths(np.random.default_rng(6491), npar, KS, m6) / s
    _, fin = endpoint_stats(pc, pe, one_sided=False)
    _, fin1 = endpoint_stats(pc, pe, one_sided=True)
    res = {
        "alpha": ALPHA,
        "mesh": m6.tolist(),
        "ks": KS.tolist(),
        "gaussian_draws": n,
        "two_sided": two,
        "one_sided_endpoints": one,
        "finite_k_pareto": {"paths": npar, "two_sided": fin, "one_sided_endpoints": fin1},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "union_endpoint_bound.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("two_sided", "one_sided_endpoints")}, indent=1))


if __name__ == "__main__":
    main()
