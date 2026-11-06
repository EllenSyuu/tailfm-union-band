"""Audit of the second-order estimator ρ̂ (FAGH 2003) used by the plug-in band.

Post hoc diagnostic (2026-11-05), development seed 6450 only. The frozen method uses
k₁ = ⌊n^0.999⌋ and switches from τ = 0 to τ = 1 whenever the τ = 0 estimate is below −1.
This script asks whether the plug-in failures come from that recipe rather than from ρ̂
itself: it reports the τ = 0 and τ = 1 estimates separately at k₁ = n^e for
e ∈ {0.999, 0.95, 0.9}, over replications, for the benchmark families and n up to 10⁶.

Usage: python -I scripts/dev_phase3_rhohat_tuning.py OUTDIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailfm.tails import draw  # noqa: E402

CASES = (("burr", -0.25), ("burr", -0.5), ("burr", -1.0), ("frechet", None), ("abs_t", None))
XI = 0.5
NS = ((5000, 40), (100_000, 20), (1_000_000, 10))
EXPS = (0.999, 0.95, 0.9)


def fagh(v: np.ndarray, tau: int) -> float:
    m1, m2, m3 = v.mean(), (v**2).mean(), (v**3).mean()
    if tau == 0:
        a, b, c = math.log(m1), 0.5 * math.log(m2 / 2), math.log(m3 / 6) / 3
    else:
        a, b, c = m1, math.sqrt(m2 / 2), (m3 / 6) ** (1 / 3)
    t = (a - b) / (b - c)
    return -abs(3 * (t - 1) / (t - 3))


def main() -> None:
    out = Path(sys.argv[1])
    res = []
    for f, (fam, rho) in enumerate(CASES):
        true = {"frechet": -1.0, "abs_t": -2 * XI}.get(fam, rho)
        for n, reps in NS:
            est = {(e, t): [] for e in EXPS for t in (0, 1)}
            for r in range(reps):
                rng = np.random.default_rng(np.random.SeedSequence([6450, f, n], spawn_key=(r,)))
                logs = np.sort(np.log(draw(fam, n, XI, rng, rho).x))[::-1]
                for e in EXPS:
                    k1 = int(n**e)
                    v = logs[:k1] - logs[k1]
                    for t in (0, 1):
                        est[(e, t)].append(fagh(v, t))
            res.append(
                {
                    "family": fam,
                    "rho": true,
                    "n": n,
                    "reps": reps,
                    "est": {
                        f"{e}|{t}": np.percentile(x, [25, 50, 75]).round(3).tolist()
                        for (e, t), x in est.items()
                    },
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "rhohat_tuning.json").write_text(json.dumps(res, indent=1))
    L = [
        "# ρ̂ audit: FAGH estimator by k₁ and τ (dev seed 6450, ξ = 0.5)",
        "",
        "Median [IQR] over replications. The frozen recipe is k₁ = n^0.999 with the switch "
        "τ = 0 → 1 when the τ = 0 estimate is < −1. Post hoc diagnostic.",
        "",
        "| family (true ρ) | n (reps) | "
        + " | ".join(f"n^{e}, τ={t}" for e in EXPS for t in (0, 1))
        + " |",
        "|---|---|" + "---|" * (2 * len(EXPS)),
    ]
    for x in res:
        cells = []
        for e in EXPS:
            for t in (0, 1):
                q = x["est"][f"{e}|{t}"]
                cells.append(f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]")
        head = f"| {x['family']} ({x['rho']}) | {x['n']} ({x['reps']}) | "
        L.append(head + " | ".join(cells) + " |")
    (out / "rhohat_tuning.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
