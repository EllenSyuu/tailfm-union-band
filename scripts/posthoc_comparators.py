"""Post hoc comparators on the confirmatory samples (added 2026-11-05, after the benchmark).

Not pre-registered. The union band and every pre-registered method are unchanged; this
script only evaluates two *additional* comparators on the same 20 × 300 samples, which are
regenerated deterministically from the spent test seed 2033 (`phase3_bench.world`). The
samples are not used to choose or tune anything in our method.

1. Sasaki & Wang (2024), basic interval H^O at a deterministic threshold k (their eq. 11
   with r = 1), at every grid k:  Hill ± (q·Hill + B̄)/√k,  q = z_{0.95} = 1.645 (90%),
   with their rule-of-thumb bias bound A = 0.1 ξ̂(1 + 2ξ̂)√k, ρ_SW = 2ξ̂, so that
   B̄/√k = A/((1 + ρ_SW)√k) = 0.1 ξ̂ (ξ̂ = Hill at the same k). Their snooping version needs
   critical values from their Table 1 and is not implemented.
2. Plug-in band with ρ̂ computed as in Caeiro, Gomes & Pestana (2005): k₁ =
   min(n − 1, ⌊2n^0.995/ln ln n⌋), ρ̂ = −|3(T − 1)/(T − 3)| with T from Fraga Alves, Gomes &
   de Haan (2003); τ = 0 if the true ρ ∈ [−1, 0), τ = 1 if ρ < −1 (CGP Remark 2.1, applied
   with knowledge of the true region, i.e. favourably to the comparator). Posterior band and
   Wald twin exactly as the frozen plug-in otherwise (`phase3_bench.intervals`).

Usage: python scripts/posthoc_comparators.py OUTDIR [--workers 2] [--reps 300]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailfm import phase3_bench as B  # noqa: E402
from tailfm.tail_estimators import hill_path  # noqa: E402

RAW = Path("reports/phase3/benchmark/raw")
_CAL: dict = {}


def cgp_rho(x: np.ndarray, rho_true: float) -> float:
    n = x.size
    k1 = min(n - 1, int(math.floor(2 * n**0.995 / math.log(math.log(n)))))
    logs = np.sort(np.log(x))[::-1]
    v = logs[:k1] - logs[k1]
    m1, m2, m3 = v.mean(), (v**2).mean(), (v**3).mean()
    if rho_true >= -1:
        a, b, c = math.log(m1), 0.5 * math.log(m2 / 2), math.log(m3 / 6) / 3
    else:
        a, b, c = m1, math.sqrt(m2 / 2), (m3 / 6) ** (1 / 3)
    t = (a - b) / (b - c)
    return min(-0.05, -abs(3 * (t - 1) / (t - 3)))


def _cal(ks):
    if ks not in _CAL:
        _CAL[ks] = B.Calibration(ks, sims_gp=10)
    return _CAL[ks]


def job(args):
    f, xi_i, n, r = args
    x, xi = B.world(2033, f, xi_i, n, r)
    fam, rb = B.FAMILIES[f]
    ks = B.grid(n)
    rho_true = B.true_rho(fam, rb, xi)
    h = hill_path(x, np.array(ks))
    sw = np.stack([h - 1.645 * h / np.sqrt(ks) - 0.1 * h, h + 1.645 * h / np.sqrt(ks) + 0.1 * h], 1)
    rho_c = cgp_rho(x, rho_true)
    cal = _cal(ks)
    post = B.intervals(x, "epd_rhohat", ks, rho_c, cal)
    wald = B.intervals(x, "wald_epd", ks, rho_c, cal)
    return {
        "r": r,
        "rho_cgp": rho_c,
        "sw": sw.round(6).tolist(),
        "cgp_post": post.round(6).tolist(),
        "cgp_wald": wald.round(6).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cells = []
    for f in range(len(B.FAMILIES)):
        for xi_i in range(len(B.XIS)):
            for n in B.NS:
                cells.append((f, xi_i, n))
    summary = []
    with Pool(a.workers) as pool:
        for f, xi_i, n in cells:
            fam, rb = B.FAMILIES[f]
            xi = B.XIS[xi_i]
            raw = json.load(gzip.open(RAW / f"cell_{fam}_{rb}_{xi}_{n}.json.gz"))
            sel = {w["r"]: w["epd_union"]["sel"] for w in raw["worlds"]}
            res = pool.map(job, [(f, xi_i, n, r) for r in range(a.reps)])
            (a.out / f"cell_{fam}_{rb}_{xi}_{n}.json").write_text(json.dumps(res))

            def cov(iv, lo_i, hi_i, xi=xi):
                return (iv[..., lo_i] <= xi) & (xi <= iv[..., hi_i])

            sw = np.array([w["sw"] for w in res])
            cp = np.array([w["cgp_post"] for w in res])
            cw = np.array([w["cgp_wald"] for w in res])
            row = {
                "family": fam,
                "rho": B.true_rho(fam, rb, xi),
                "xi": xi,
                "n": n,
                "reps": a.reps,
                "in_assumption": not (fam == "burr" and rb == -0.25),
                "rho_cgp_median": float(np.median([w["rho_cgp"] for w in res])),
                "rho_cgp_iqr": np.percentile([w["rho_cgp"] for w in res], [25, 75]).tolist(),
            }
            c_sw = cov(sw, 0, 1)
            row["sw_pointwise"] = c_sw.mean(0).tolist()
            row["sw_width"] = (sw[..., 1] - sw[..., 0]).mean(0).tolist()
            for rule in ("samsee", "guillou_hall", "largest"):
                j = np.array([sel[w["r"]][rule] for w in res])
                row[f"sw_after_{rule}"] = float(c_sw[np.arange(len(res)), j].mean())
            for name, iv in (("cgp_post", cp), ("cgp_wald", cw)):
                band = cov(iv, 3, 4)
                row[f"{name}_simultaneous"] = float(band.all(1).mean())
                row[f"{name}_after"] = {
                    rule: float(
                        band[np.arange(len(res)), np.array([sel[w["r"]][rule] for w in res])].mean()
                    )
                    for rule in ("samsee", "guillou_hall", "largest")
                }
                row[f"{name}_bandwidth_top"] = float((iv[:, -1, 4] - iv[:, -1, 3]).mean())
            summary.append(row)
            print(fam, rb, xi, n, "done", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
