"""Exact limiting coverage and width of the finite-mesh union band (preprint Theorem 2 and Remark 2).

Under (A1)–(A2) with √k_max δ(U(n/k_max)) → λ and every grid point in regime R1, the
centred EPD estimators at the mesh values ρ_m satisfy, jointly in (j, m),

    √k_j (ξ̂_j(ρ_m) − ξ)/ξ  ⇒  G_jm + λ_j β(ρ, ρ_m)/ξ,   λ_j = λ t_j^{1/2−ρ},

where G is the Gaussian limit under exact Pareto tails (Le Cam's third lemma; Lemma 1).
The union band at grid point j covers ξ iff

    min_m {G_jm + λ_j β_m − c_m s_m} ≤ 0 ≤ max_m {G_jm + λ_j β_m + c_m s_m},

with s_m = (1 − ρ_m)/|ρ_m| and c_m the Pareto calibration of the band at ρ_m. So the
limiting simultaneous coverage Γ_M(ρ, λ) is an explicit Gaussian probability. This script
evaluates it by Monte Carlo from the exact Gaussian limit G, whose covariance (preprint eq. 3.1)
is computed in closed form by `limit_cov` (revised 2026-11-06; the earlier version simulated the
estimator from Pareto log-spacings at k ≤ 1024, which adds finite-k skewness). `g_paths` (the
finite-k Pareto simulation) is kept for the prediction check, which wants finite-k behaviour.
(ξ = 1 without loss of generality: all quantities are scale-free; λ is in units of ξ.)

Usage: python -I scripts/theory_union_limit.py OUTDIR [--sims 8000] [--kmax 1024]
Seeds: calibration 6460, evaluation 6461 (development namespace; no test seed is used).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ALPHA = 0.1
M_GRID = 5


def mesh(npts: int) -> np.ndarray:
    return -np.exp(np.linspace(math.log(0.5), math.log(2.0), npts))


def beta(rho: float, rp: np.ndarray) -> np.ndarray:
    """Lemma 1: asymptotic bias factor of the EPD estimator at ρ′ when the truth is ρ."""
    return rho * (1 - rp) * (rp - rho) / (rp * (1 - rho) * (1 - rho - rp))


def g_paths(rng: np.random.Generator, sims: int, ks: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """G[s, j, m] = √k_j (ξ̂_j(ρ_m) − 1) under exact Pareto(ξ = 1), BJS estimator."""
    kmax = int(ks[-1])
    out = np.empty((sims, ks.size, rhos.size))
    chunk = 500
    for a in range(0, sims, chunk):
        b = min(sims, a + chunk)
        e = rng.exponential(size=(b - a, kmax))
        for j, k in enumerate(ks):
            w = e[:, :k] / np.arange(1, k + 1)
            v = np.cumsum(w[:, ::-1], axis=1)[:, ::-1]  # log-excesses over X_(n−k), Rényi
            h = v.mean(axis=1)
            for m, r in enumerate(rhos):
                ee = np.exp(r / h[:, None] * v).mean(axis=1)
                delta = h * (1 - 2 * r) * (1 - r) ** 3 * r**-4 * (ee - 1 / (1 - r))
                out[a:b, j, m] = math.sqrt(k) * (h - delta * r / (1 - r) - 1.0)
    return out


def _cov_psi(rl: float, rk: float, d: float) -> float:
    """cov(psi_rl(E), psi_rk(E + d)) for E ~ Exp(1), d >= 0 (closed form)."""

    def coef(r):
        return (1 - r) ** 2 / r**2, (1 - 2 * r) * (1 - r) / r**3

    def cov_e(s):  # cov(E, e^{sE})
        return s / (1 - s) ** 2

    def cov_ee(s, u):  # cov(e^{sE}, e^{uE})
        return 1 / (1 - s - u) - 1 / ((1 - s) * (1 - u))

    a1, b1 = coef(rl)
    a2, b2 = coef(rk)
    g = (1 - rk) * math.exp(rk * d)
    return (
        a1 * a2
        - a1 * b2 * g * cov_e(rk)
        - b1 * (1 - rl) * a2 * cov_e(rl)
        + b1 * b2 * (1 - rl) * g * cov_ee(rl, rk)
    )


def limit_cov(t: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """Covariance of the limit array G (preprint eq. 3.1), flattened as (j, l) -> j * L + l."""
    m, L = len(t), len(rhos)
    S = np.zeros((m * L, m * L))
    for j in range(m):
        for jj in range(m):
            for a in range(L):
                for b in range(L):
                    if t[j] <= t[jj]:
                        d = math.log(t[jj] / t[j])
                        v = math.sqrt(t[j] / t[jj]) * _cov_psi(rhos[a], rhos[b], d)
                    else:
                        d = math.log(t[j] / t[jj])
                        v = math.sqrt(t[jj] / t[j]) * _cov_psi(rhos[b], rhos[a], d)
                    S[j * L + a, jj * L + b] = v
    return S


def g_gauss(rng: np.random.Generator, sims: int, ks: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """Draws of the exact Gaussian limit G[s, j, m] (no finite-k effects)."""
    t = ks / ks[-1]
    w, v = np.linalg.eigh(limit_cov(t, rhos))
    root = v * np.sqrt(np.clip(w, 0, None))
    z = rng.standard_normal((sims, ks.size * rhos.size)) @ root.T
    return z.reshape(sims, ks.size, rhos.size)


def calibrate(g: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    s = (1 - rhos) / np.abs(rhos)
    stat = np.max(np.abs(g) / s[None, None, :], axis=1)  # (sims, m)
    return np.quantile(stat, 1 - ALPHA, axis=0)


def coverage_width(g, rhos, c, t, rho, lam):
    s = (1 - rhos) / np.abs(rhos)
    lamj = lam * t ** (0.5 - rho)
    shift = lamj[:, None] * beta(rho, rhos)[None, :]  # (j, m)
    lo = (g + shift[None] - (c * s)[None, None]).min(axis=2)
    hi = (g + shift[None] + (c * s)[None, None]).max(axis=2)
    cov = np.mean(np.all((lo <= 0) & (hi >= 0), axis=1))
    return float(cov), float(np.mean(hi[:, -1] - lo[:, -1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sims", type=int, default=8000)
    ap.add_argument("--kmax", type=int, default=1024)
    a = ap.parse_args()
    ks = np.array(sorted(a.kmax // 2**j for j in range(M_GRID)))
    t = ks / ks[-1]
    meshes = {n: mesh(n) for n in (6, 12, 24)}
    allr = np.unique(np.concatenate(list(meshes.values())).round(12))
    g_cal = g_gauss(np.random.default_rng(6460), a.sims // 2, ks, allr)
    g_ev = g_gauss(np.random.default_rng(6461), a.sims, ks, allr)
    c_all = calibrate(g_cal, allr)
    idx = {n: np.searchsorted(allr, np.sort(m).round(12)) for n, m in meshes.items()}
    for n, ii in idx.items():  # each mesh must be recovered exactly
        assert np.allclose(allr[ii], np.sort(meshes[n]))
    rho_eval = np.round(
        np.concatenate(
            [
                -np.exp(np.linspace(math.log(0.5), math.log(2), 61)),
                [-0.25, -0.35, -0.45, -2.5, -3.0],
            ]
        ),
        4,
    )
    lams = np.round(np.union1d(np.linspace(-12, 12, 49), [-48, -32, -24, -16, 16, 24, 32, 48]), 3)
    # oracle (known ρ) width 2 c(ρ) s(ρ): c interpolated in log|ρ| over the 24-point mesh
    lr24 = np.log(-allr[idx[24]])[::-1]  # increasing, as np.interp requires
    c24 = c_all[idx[24]][::-1]
    res = {"ks": ks.tolist(), "sims": a.sims, "alpha": ALPHA, "meshes": {}}
    for n, ii in idx.items():
        rows = []
        for rho in rho_eval:
            covs, wids = [], []
            for lam in lams:
                cv, wd = coverage_width(g_ev[:, :, ii], allr[ii], c_all[ii], t, rho, lam)
                covs.append(cv)
                wids.append(wd)
            co = float(np.interp(math.log(-rho), lr24, c24))
            oracle = 2 * co * (1 - rho) / abs(rho)
            rows.append(
                {"rho": float(rho), "coverage": covs, "width_top": wids, "oracle_width_top": oracle}
            )
        res["meshes"][str(n)] = {
            "rho_mesh": allr[ii].tolist(),
            "c": c_all[ii].tolist(),
            "rows": rows,
        }
    res["lambda"] = lams.tolist()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "union_limit.json").write_text(json.dumps(res))
    se = math.sqrt(0.9 * 0.1 / a.sims)
    L = [
        "# Limiting coverage and width of the finite-mesh union band",
        "",
        f"`scripts/theory_union_limit.py`; Monte Carlo evaluation from "
        f"the exact Gaussian limit at t_j = k_j/k_max for k = {ks.tolist()}, "
        f"{a.sims} evaluation paths "
        f"(seed 6461), calibration on {a.sims // 2} independent paths (seed 6460). Nominal "
        f"0.90; Monte Carlo s.e. of a coverage near 0.9 ≈ {se:.4f}. λ is the local bias "
        "√k δ(U(n/k)) at the top grid point, in units of ξ, swept over [−48, 48] "
        "(fine on [−12, 12]).",
        "",
        "| mesh | ρ range | worst coverage over λ | at (ρ, λ) | coverage at λ = 0 (min over ρ) "
        "| width/oracle at top, λ = 0 (max over ρ) | width/oracle at top, worst λ |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in (6, 12, 24):
        rows = res["meshes"][str(n)]["rows"]
        for name, sel in (
            ("ρ ∈ R = [−2, −0.5]", lambda r: -2 - 1e-9 <= r <= -0.5 + 1e-9),
            ("ρ ∉ R", lambda r: not (-2 - 1e-9 <= r <= -0.5 + 1e-9)),
        ):
            rr = [r for r in rows if sel(r["rho"])]
            worst = min(
                (min(r["coverage"]), r["rho"], lams[int(np.argmin(r["coverage"]))]) for r in rr
            )
            i0 = int(np.where(lams == 0)[0][0])
            c0 = min(r["coverage"][i0] for r in rr)
            w0 = max(r["width_top"][i0] / r["oracle_width_top"] for r in rr)
            wmax = max(max(r["width_top"]) / r["oracle_width_top"] for r in rr)
            L.append(
                f"| {n} | {name} | {worst[0]:.3f} | ({worst[1]:.3f}, {worst[2]:+.1f}) "
                f"| {c0:.3f} | {w0:.2f} | {wmax:.2f} |"
            )
    L += [
        "",
        "Per-ρ worst case, 6-point mesh (the evaluated method):",
        "",
        "| ρ | worst coverage | λ at worst | coverage at λ = −48 / +48 |",
        "|---|---|---|---|",
    ]
    for r in res["meshes"]["6"]["rows"][::6] + res["meshes"]["6"]["rows"][61:]:
        cv = r["coverage"]
        L.append(
            f"| {r['rho']:.3f} | {min(cv):.3f} | {lams[int(np.argmin(cv))]:+.1f} "
            f"| {cv[0]:.3f} / {cv[-1]:.3f} |"
        )
    (a.out / "union_limit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
