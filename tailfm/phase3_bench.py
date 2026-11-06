"""Phase 3 confirmatory benchmark (protocol: docs/phase-3-benchmark.md).

Frozen method (selection-valid credible bands for ξ):
* grid: k_max = ⌊n^0.75⌋, k_j = ⌊k_max / 2^j⌋, j = 0..4 (five points, ascending);
* second-order (EPD) tail posterior with ρ̂ (FAGH at ⌊n^0.999⌋, capped at −0.05) — primary;
* robust variant: union over ρ ∈ R = [−2, −0.5] (6 log-spaced points) of known-ρ bands;
* band: central posterior interval at level 1 − α*, α* from the exact Pareto calibration of
  the BJS estimator path at ρ (tailfm.phase3.calibrate_level), α = 0.1.
Comparators: first-order (Pareto) posterior with its Rényi-calibrated band; GP posterior
(Dombry, Padoan & Rizzelli) with a pivotal band calibrated at ξ̂ from a precomputed table
(tailfm.phase3.calibrate_level_gp, interpolated in ξ); frequentist twin (BJS estimator ±
c·se at ρ̂). Selection rules: SAMSEE, Guillou–Hall, Lepski-type stability on each method's
own intervals, largest grid k (all mapped to the nearest grid point in log k).
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import phase3 as P3
from . import selectors_classical as SC
from .selectors import samsee
from .tail_estimators import epd_path, second_order_estimates
from .tails import draw

ALPHA = 0.1
GRID_EXP = 0.75
M = 5
R_GRID = -np.exp(np.linspace(np.log(0.5), np.log(2.0), 6))
GP_XI_TABLE = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
POST = dict(nx=161, nd=41)
METHODS = ("pareto", "gp", "epd_rhohat", "epd_union", "wald_epd")
RULES = ("samsee", "guillou_hall", "lepski", "largest")
# (family, ρ for Burr) — in-assumption cells have ρ ≤ −0.5
FAMILIES = (
    ("burr", -0.5),
    ("burr", -1.0),
    ("frechet", None),
    ("abs_t", None),
    ("burr", -0.25),  # outside the assumption ρ ≤ −0.5: reported, not in the decision rules
)
XIS = (0.5, 1.0)
NS = (2000, 5000)


def grid(n: int) -> tuple[int, ...]:
    kmax = int(math.floor(n**GRID_EXP))
    return tuple(sorted(kmax // 2**j for j in range(M)))


def true_rho(fam: str, rho: float | None, xi: float) -> float:
    return {"frechet": -1.0, "abs_t": -2.0 * xi}.get(fam, rho)


class Calibration:
    """Calibrations that depend only on the grid (computed once per n, calibration seeds)."""

    def __init__(self, ks: tuple[int, ...], sims_gp: int = 600):
        self.ks = ks
        self.pareto = P3.calibrate_level(ks, None)[0]
        self.union = [P3.calibrate_level(ks, float(g), sims=1500)[0] for g in R_GRID]
        gp = [P3.calibrate_level_gp(ks, xi, sims=sims_gp)[1] for xi in GP_XI_TABLE]
        self.gp_c = np.array(gp)
        self._epd_cache: dict[float, tuple[float, float]] = {}

    def epd(self, rho: float) -> tuple[float, float]:
        key = round(rho, 3)
        if key not in self._epd_cache:
            self._epd_cache[key] = P3.calibrate_level(self.ks, key, sims=1500)
        return self._epd_cache[key]

    def gp(self, xi_hat: float) -> float:
        lx = np.log(np.clip(xi_hat, GP_XI_TABLE[0], GP_XI_TABLE[-1]))
        c = float(np.interp(lx, np.log(GP_XI_TABLE), self.gp_c))
        return float(2 * P3.stats.norm.sf(c))


def _iv(lx, cdf, a_star):
    return np.concatenate(
        [
            P3.quantiles(lx, cdf, [0.05, 0.5, 0.95]),
            P3.quantiles(lx, cdf, [a_star / 2, 1 - a_star / 2]),
        ]
    )


def intervals(x: np.ndarray, method: str, ks, rho_hat: float, cal: Calibration) -> np.ndarray:
    """(m, 5): lo90, median, hi90, band_lo, band_hi at every grid k."""
    rows = []
    if method == "gp":
        h = float(np.mean(np.log(np.sort(x)[-ks[-1] :]) - np.log(np.sort(x)[-ks[-1] - 1])))
        a_gp = cal.gp(h)
    for k in ks:
        if method == "pareto":
            rows.append(_iv(*P3.pareto_tail_posterior(x, k), cal.pareto))
        elif method == "gp":
            rows.append(_iv(*P3.gp_tail_posterior(x, k, ng=161, ns=81), a_gp))
        elif method == "epd_rhohat":
            rows.append(_iv(*P3.epd_tail_posterior(x, k, rho_hat, **POST), cal.epd(rho_hat)[0]))
        elif method == "epd_union":
            parts = np.array(
                [
                    _iv(*P3.epd_tail_posterior(x, k, float(g), **POST), cal.union[i])
                    for i, g in enumerate(R_GRID)
                ]
            )
            rows.append(
                np.array(
                    [
                        parts[:, 0].min(),
                        np.median(parts[:, 1]),
                        parts[:, 2].max(),
                        parts[:, 3].min(),
                        parts[:, 4].max(),
                    ]
                )
            )
        elif method == "wald_epd":
            est = float(epd_path(x, np.array([k]), rho=rho_hat)[0])
            sd = abs(est) * (1 - rho_hat) / abs(rho_hat) / math.sqrt(k)
            c = cal.epd(rho_hat)[1]
            rows.append(
                np.array([est - 1.645 * sd, est, est + 1.645 * sd, est - c * sd, est + c * sd])
            )
    return np.array(rows)


def lepski(iv: np.ndarray) -> int:
    best = 0
    for j in range(iv.shape[0]):
        if all(iv[j, 0] <= iv[i, 4] and iv[j, 2] >= iv[i, 3] for i in range(j)):
            best = j
        else:
            break
    return best


def nearest(ks, k: int) -> int:
    return int(np.argmin(np.abs(np.log(np.array(ks)) - math.log(max(2, k)))))


def evaluate(x: np.ndarray, xi: float, ks, cal: Calibration) -> dict:
    rho_hat = min(-0.05, second_order_estimates(x)[0])
    fixed = {
        "samsee": nearest(ks, samsee(x)["k"]),
        "guillou_hall": nearest(ks, SC.guillou_hall(x)["k"]),
        "largest": len(ks) - 1,
    }
    out = {"rho_hat": rho_hat}
    for m in METHODS:
        t0 = time.perf_counter()
        iv = intervals(x, m, ks, rho_hat, cal)
        sec = time.perf_counter() - t0
        sel = {**fixed, "lepski": lepski(iv)}
        out[m] = {"iv": iv.round(6).tolist(), "sel": sel, "seconds": sec}
    return out


def world(seed: int, f: int, xi_i: int, n: int, r: int):
    fam, rho = FAMILIES[f]
    xi = XIS[xi_i]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8, f, xi_i, n], spawn_key=(r,)))
    return draw(fam, n, xi, rng, rho).x, xi
