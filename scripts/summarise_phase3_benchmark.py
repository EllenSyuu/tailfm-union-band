"""Summaries and pre-registered decision rules of the Phase 3 benchmark
(docs/phase-3-benchmark.md). Uses only json and numpy, reading the raw cells.

Usage: python -I scripts/summarise_phase3_benchmark.py RAWDIR OUT.md OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

METHODS = ("pareto", "gp", "epd_rhohat", "epd_union", "wald_epd")
RULES = ("samsee", "guillou_hall", "lepski", "largest")
LABEL = {
    "pareto": "first-order",
    "gp": "GP (pivotal band)",
    "epd_rhohat": "second-order ρ̂",
    "epd_union": "second-order union ρ∈[−2,−0.5]",
    "wald_epd": "Wald twin",
}


def cell_summary(c):
    xi = c["xi"]
    res = {}
    for m in METHODS:
        iv = np.array([w[m]["iv"] for w in c["worlds"]])  # (R, K, 5)
        pc = (iv[:, :, 0] <= xi) & (xi <= iv[:, :, 2])
        bc = (iv[:, :, 3] <= xi) & (xi <= iv[:, :, 4])
        after = {}
        for q in RULES:
            j = np.array([w[m]["sel"][q] for w in c["worlds"]])
            idx = np.arange(len(j))
            after[q] = {"interval": float(pc[idx, j].mean()), "band": float(bc[idx, j].mean())}
        res[m] = {
            "simultaneous": float(bc.all(1).mean()),
            "pointwise_by_k": pc.mean(0).round(4).tolist(),
            "after": after,
            "width90_top": float((iv[:, -1, 2] - iv[:, -1, 0]).mean()),
            "widthband_top": float((iv[:, -1, 4] - iv[:, -1, 3]).mean()),
            "bias_by_k": (iv[:, :, 1].mean(0) - xi).round(4).tolist(),
            "seconds": float(np.mean([w[m]["seconds"] for w in c["worlds"]])),
        }
    return res


def main(raw, out_md, out_json):
    cells = [json.loads(p.read_text()) for p in sorted(Path(raw).glob("cell_*.json"))]
    rows = []
    for c in cells:
        rows.append(
            {
                "family": c["family"],
                "rho": c["rho"],
                "xi": c["xi"],
                "n": c["n"],
                "reps": len(c["worlds"]),
                "in_assumption": c["rho"] <= -0.5,
                **cell_summary(c),
            }
        )
    ins = [r for r in rows if r["in_assumption"]]
    d1a = sum(all(r["epd_rhohat"]["after"][q]["band"] >= 0.85 for q in RULES) for r in ins)
    d1b = sum(all(r["epd_union"]["after"][q]["band"] >= 0.85 for q in RULES) for r in ins)
    d2 = sum(r["pareto"]["after"]["lepski"]["interval"] < 0.80 for r in ins)
    d3 = sum(
        abs(r["epd_rhohat"]["simultaneous"] - 0.9) < abs(r["wald_epd"]["simultaneous"] - 0.9)
        for r in ins
    )
    n_in = len(ins)
    dec = {
        "cells_in_assumption": n_in,
        "D1a_rhohat_band_after_selection_ge_0.85_all_rules": d1a,
        "D1a_pass": d1a >= n_in - 2,
        "D1b_union_band_after_selection_ge_0.85_all_rules": d1b,
        "D1b_pass": d1b == n_in,
        "D2_first_order_lepski_interval_lt_0.80": d2,
        "D2_pass": d2 >= n_in // 2,
        "D3_bayes_band_closer_to_nominal_than_wald": d3,
        "D3_pass": d3 >= int(np.ceil(0.625 * n_in)),
        "D4_narrower_than_gp_with_both_valid": sum(
            r["epd_rhohat"]["width90_top"] < r["gp"]["width90_top"]
            and r["epd_rhohat"]["simultaneous"] >= 0.85
            and r["gp"]["simultaneous"] >= 0.85
            for r in ins
        ),
    }
    dec["main_claims_supported"] = bool(dec["D1a_pass"] and dec["D1b_pass"] and dec["D2_pass"])
    Path(out_json).write_text(json.dumps({"decisions": dec, "rows": rows}, indent=1) + "\n")
    L = ["# Phase 3 benchmark — summary", "", "Decisions (pre-registered):", ""]
    L += [f"- {k}: {v}" for k, v in dec.items()]
    L += [
        "",
        "| family | ρ | ξ | n | method | simultaneous | band after k̂: SAMSEE/GH/Lepski/largest "
        "| interval after k̂: SAMSEE/GH/Lepski/largest | width top: interval/band |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for m in METHODS:
            v = r[m]
            b = "/".join(f"{v['after'][q]['band']:.2f}" for q in RULES)
            i = "/".join(f"{v['after'][q]['interval']:.2f}" for q in RULES)
            L.append(
                f"| {r['family']} | {r['rho']} | {r['xi']} | {r['n']} | {LABEL[m]} | "
                f"{v['simultaneous']:.3f} | {b} | {i} | "
                f"{v['width90_top']:.3f}/{v['widthband_top']:.3f} |"
            )
    Path(out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
