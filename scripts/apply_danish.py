"""Phase 3 application: Danish fire insurance losses (McNeil 1997), from R package evir.

The data are not committed. This script clones the public CRAN mirror github.com/cran/evir
into a fresh temporary directory (or uses --evir PATH), reads data/danish.rda with pyreadr
(requires `pyreadr` and `pandas`), and applies the *frozen* Phase 3 method of
tailfm/phase3_bench.py unchanged: grid K = grid(n) (k_max = ⌊n^0.75⌋, five points), the
same five methods (first-order, GP with pivotal band, second-order with ρ̂, union over
ρ ∈ [−2, −0.5], Wald twin) and the same selection rules (SAMSEE, Guillou–Hall,
Lepski-type, largest k).

Usage: python scripts/apply_danish.py OUT_DIR [--evir PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tailfm import phase3_bench as B
from tailfm import selectors_classical as SC
from tailfm.selectors import samsee
from tailfm.tail_estimators import second_order_estimates


def load(evir: str | None) -> np.ndarray:
    import pyreadr

    if evir is None:
        tmp = tempfile.mkdtemp(prefix="evir_")
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "https://github.com/cran/evir", tmp], check=True
        )
        evir = tmp
    r = pyreadr.read_r(str(Path(evir) / "data" / "danish.rda"))
    return np.asarray(next(iter(r.values())).iloc[:, 0], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--evir", default=None)
    a = ap.parse_args()
    x = load(a.evir)
    n = x.size
    ks = B.grid(n)
    cal = B.Calibration(ks)
    res = B.evaluate(x, float("nan"), ks, cal)
    raw_k = {"samsee": int(samsee(x)["k"]), "guillou_hall": int(SC.guillou_hall(x)["k"])}
    _, beta_hat = second_order_estimates(x)
    out = {
        "n": n,
        "ks": ks,
        "rho_hat": res["rho_hat"],
        "beta_hat": beta_hat,
        "selector_k_raw": raw_k,
        "methods": {m: {"iv": res[m]["iv"], "sel": res[m]["sel"]} for m in B.METHODS},
    }
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "danish.json").write_text(json.dumps(out, indent=1) + "\n")
    L = [
        f"# Danish fire losses (n = {n}) — frozen Phase 3 method",
        "",
        f"Grid K = {ks} (k_max = ⌊n^0.75⌋). ρ̂ = {res['rho_hat']:.3f} (FAGH at ⌊n^0.999⌋). "
        f"Raw selector k: SAMSEE {raw_k['samsee']}, Guillou–Hall {raw_k['guillou_hall']} "
        "(mapped to the nearest grid point in log k).",
        "",
        "Each cell: posterior median (estimate for the Wald twin) [90% interval] {band}.",
        "",
        "| k | " + " | ".join(B.METHODS) + " |",
        "|---|" + "---|" * len(B.METHODS),
    ]
    for j, k in enumerate(ks):
        cells = []
        for m in B.METHODS:
            v = res[m]["iv"][j]
            cells.append(f"{v[1]:.3f} [{v[0]:.3f}, {v[2]:.3f}] {{{v[3]:.3f}, {v[4]:.3f}}}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "Selected grid index per method (SAMSEE / Guillou–Hall / Lepski / largest):", ""]
    for m in B.METHODS:
        s = res[m]["sel"]
        L.append(f"- {m}: " + " / ".join(f"k={ks[s[q]]}" for q in B.RULES))
    (o / "danish.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
