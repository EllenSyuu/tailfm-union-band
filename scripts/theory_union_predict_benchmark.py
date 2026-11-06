"""Predict the confirmatory union-band coverage from the limit theory (preprint Section 6.3).

Post hoc theory check, added 2026-11-05 after the confirmatory benchmark. It uses no new
data from any test seed: it compares the published per-cell simultaneous coverage of the
union band (reports/phase3/benchmark/summary.json, seed 2033, 300 reps) with the
coverage predicted by the first-order limit

    P( for all j: min_m {G_jm + λ_j β(ρ, ρ_m) − c_m s_m} ≤ 0 ≤ max_m {…+ c_m s_m} ),

where G is simulated exactly under Pareto tails at the cell's own grid (Rényi, BJS
estimator, development seeds 6470/6471), c_m are calibrated on independent paths, and
λ_j = √k_j δ(U(n/k_j))/ξ is computed exactly from the family's survival function through
(A1): ξ⁻¹δ(x) = F̄(x) x^{1/ξ}/C − 1, C = lim F̄(x) x^{1/ξ}. The prediction ignores
third-order terms, so a gap at large |λ_j| is informative, not a failure of the check.

Usage: python -I scripts/theory_union_predict_benchmark.py OUTDIR [--sims 6000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailfm.phase3_bench import R_GRID, grid  # noqa: E402
from tailfm.tails import survival  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from theory_union_limit import beta, calibrate, g_paths  # noqa: E402

FAM_RHO = {"burr": None, "frechet": -1.0, "abs_t": None}


def lam_path(fam: str, rho: float | None, xi: float, n: int, ks) -> np.ndarray:
    def sf(x):
        return float(survival(fam, np.array([x]), xi, rho)[0])

    big = 1e12 if fam != "abs_t" else 1e8
    c = sf(big) * big ** (1 / xi)
    out = []
    for k in ks:
        p = k / n
        lp = math.log(p)
        u = optimize.brentq(lambda x, lp=lp: math.log(sf(x)) - lp, 1e-6, 1e40)
        d_over_xi = sf(u) * u ** (1 / xi) / c - 1.0
        out.append(math.sqrt(k) * d_over_xi)  # = √k δ(U(n/k)) / ξ
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sims", type=int, default=6000)
    a = ap.parse_args()
    rows = json.loads(Path("reports/phase3/benchmark/summary.json").read_text())["rows"]
    rhos = np.sort(np.asarray(R_GRID, dtype=float))
    s = (1 - rhos) / np.abs(rhos)
    cache = {}
    res = []
    for r in rows:
        n, xi, fam = r["n"], r["xi"], r["family"]
        ks = np.array(grid(n))
        if n not in cache:
            g_cal = g_paths(np.random.default_rng([6470, n]), a.sims // 2, ks, rhos)
            g_ev = g_paths(np.random.default_rng([6471, n]), a.sims, ks, rhos)
            cache[n] = (g_ev, calibrate(g_cal, rhos))
        g_ev, c = cache[n]
        rho_true = r["rho"]
        lam = lam_path(fam, r["rho"] if fam == "burr" else None, xi, n, ks)
        shift = lam[:, None] * beta(rho_true, rhos)[None, :]
        lo = (g_ev + shift[None] - (c * s)[None, None]).min(axis=2)
        hi = (g_ev + shift[None] + (c * s)[None, None]).max(axis=2)
        pred = float(np.mean(np.all((lo <= 0) & (hi >= 0), axis=1)))
        obs = r["epd_union"]["simultaneous"]
        res.append(
            {
                "family": fam,
                "rho": rho_true,
                "xi": xi,
                "n": n,
                "ks": ks.tolist(),
                "lambda_over_xi": lam.round(3).tolist(),
                "predicted": pred,
                "observed": obs,
                "in_assumption": r["in_assumption"],
            }
        )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "union_predict_benchmark.json").write_text(json.dumps(res, indent=1))
    se_o = math.sqrt(0.9 * 0.1 / 300)
    L = [
        "# Union band: predicted (limit theory) vs observed (confirmatory) coverage",
        "",
        "Post hoc check (2026-11-05), no new test-seed data. Observed: simultaneous "
        "coverage of the 6-point union band, seed 2033, 300 reps (s.e. ≈ "
        f"{se_o:.3f} near 0.9). Predicted: first-order limit with exact λ_j and exact "
        f"Pareto G at the cell's grid ({a.sims} paths). Nominal 0.90.",
        "",
        "| family | ρ | ξ | n | λ_j/ξ at k_max | predicted | observed | obs − pred |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in res:
        flag = "" if x["in_assumption"] else " (outside R)"
        L.append(
            f"| {x['family']}{flag} | {x['rho']} | {x['xi']} | {x['n']} | "
            f"{x['lambda_over_xi'][-1]:+.2f} | {x['predicted']:.3f} | {x['observed']:.3f} "
            f"| {x['observed'] - x['predicted']:+.3f} |"
        )
    (a.out / "union_predict_benchmark.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
