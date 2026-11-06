"""Post hoc: Sasaki & Wang (2024) snooping interval H^S on the confirmatory samples.

Added 2026-11-05 after the benchmark (not pre-registered). H^S (their eq. 15) is the
intersection over thresholds k_j in [r̲ k̄, k̄] of H^O(n, k_j) = Hill_j ± (q ξ̂_j + B̄)/√k_j, with
their rule-of-thumb bias bound A = 0.1 ξ̂(1 + 2ξ̂)√(r k̄), ρ_SW = 2ξ̂, B̄ = A/(1 + ρ_SW). We read
r k̄ as k_j (the threshold at which the interval is computed), which makes the bias allowance
0.1 ξ̂_j at every k_j; their text writes √(r k̄) without fixing r when k varies, so this is our
reading. q = 2.67 is their Table 1 value for β = 0.10 and r̲ = 1/20, conservative for our grid
(k_1/k_max = 1/16, five points instead of a continuum). The intersection is taken over our
five grid points with k̄ = k_max. Samples are regenerated from the spent seed 2033 exactly as
in the benchmark; nothing in our method is changed.

Usage: python scripts/posthoc_sasaki_wang_snooping.py OUTDIR
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailfm import phase3_bench as B  # noqa: E402
from tailfm.tail_estimators import hill_path  # noqa: E402

Q_SNOOP = 2.67
RAW = Path("reports/phase3/benchmark/raw")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, (fam, rb) in enumerate(B.FAMILIES):
        for xi_i, xi in enumerate(B.XIS):
            for n in B.NS:
                ks = np.array(B.grid(n))
                raw = json.load(gzip.open(RAW / f"cell_{fam}_{rb}_{xi}_{n}.json.gz"))
                cov, wid, uni_iw, emp = [], [], [], []
                for w in raw["worlds"]:
                    x, _ = B.world(2033, f, xi_i, n, w["r"])
                    h = hill_path(x, ks)
                    lo = h - Q_SNOOP * h / np.sqrt(ks) - 0.1 * h
                    hi = h + Q_SNOOP * h / np.sqrt(ks) + 0.1 * h
                    a, b = lo.max(), hi.min()
                    emp.append(a > b)
                    cov.append(bool(a <= xi <= b))
                    wid.append(max(0.0, b - a))
                    u = np.array(w["epd_union"]["iv"])
                    uni_iw.append(max(0.0, u[:, 4].min() - u[:, 3].max()))
                rows.append(
                    {
                        "family": fam,
                        "rho": B.true_rho(fam, rb, xi),
                        "xi": xi,
                        "n": n,
                        "in_assumption": not (fam == "burr" and rb == -0.25),
                        "sw_snoop_coverage": float(np.mean(cov)),
                        "sw_snoop_empty": float(np.mean(emp)),
                        "sw_snoop_width_median": float(np.median(wid)),
                        "union_intersection_width_median": float(np.median(uni_iw)),
                    }
                )
                print(fam, rb, xi, n, rows[-1]["sw_snoop_coverage"], flush=True)
    (out / "sw_snoop_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
