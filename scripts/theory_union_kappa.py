"""Least-favourable rescaling of the union critical values (preprint Section 8, limitation 6; not adopted).

For κ ∈ {1, 0.95, 0.90, 0.88, 0.86, 0.84}, evaluates inf over ρ ∈ R (31 log-spaced points)
and λ/ξ ∈ [−48, 48] of the limiting coverage Γ_M when every c_l is replaced by κ c_l.
6-point mesh, grid t_j = 2^−j, exact Gaussian limit (20000 + 20000 draws), dev seeds 6480
(calibration) / 6481.

Usage: python -I scripts/theory_union_kappa.py OUTDIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from theory_union_limit import beta, calibrate, g_gauss, mesh  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    ks = np.array([64, 128, 256, 512, 1024])
    t = ks / ks[-1]
    rh = np.sort(mesh(6))
    s = (1 - rh) / np.abs(rh)
    c = calibrate(g_gauss(np.random.default_rng(6480), 20000, ks, rh), rh)
    ge = g_gauss(np.random.default_rng(6481), 20000, ks, rh)
    rhos = -np.exp(np.linspace(math.log(0.5), math.log(2), 31))
    lams = np.union1d(np.linspace(-12, 12, 49), [-48, -32, -24, -16, 16, 24, 32, 48])
    res = []
    for kap in (1.0, 0.95, 0.9, 0.88, 0.86, 0.84):
        worst = 1.0
        for r in rhos:
            for lam in lams:
                sh = (lam * t ** (0.5 - r))[:, None] * beta(r, rh)[None, :]
                lo = (ge + sh - kap * c * s).min(2)
                hi = (ge + sh + kap * c * s).max(2)
                worst = min(worst, float(np.mean(np.all((lo <= 0) & (hi >= 0), 1))))
        res.append({"kappa": kap, "worst_coverage": worst})
        print(kap, round(worst, 3), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "union_kappa.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
