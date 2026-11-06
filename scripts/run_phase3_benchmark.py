"""Run the Phase 3 benchmark cells (protocol docs/phase-3-benchmark.md), resumable.

One JSON file per (family, ξ, n) cell; existing cells are skipped. Calibrations are computed
once per n (calibration seeds ≥ 6500, disjoint from the evaluation seed) and stored in
OUTDIR/calibration_n{n}.json. Source hashes are pinned in OUTDIR/sources.json; the runner
refuses to continue if any source file changed since the run started.

Usage: python scripts/run_phase3_benchmark.py OUTDIR --seed SEED --reps R [--workers 2]
       [--cells i,j,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tailfm import phase3_bench as B

SOURCES = (
    "tailfm/phase3_bench.py",
    "tailfm/phase3.py",
    "tailfm/tail_estimators.py",
    "tailfm/selectors.py",
    "tailfm/selectors_classical.py",
    "tailfm/tails.py",
    "scripts/run_phase3_benchmark.py",
)
_CAL = {}


def _init(cal):
    _CAL.update(cal)


def _job(args):
    seed, f, xi_i, n, r = args
    x, xi = B.world(seed, f, xi_i, n, r)
    return {"r": r, **B.evaluate(x, xi, B.grid(n), _CAL[n])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cells", default=None)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}
    meta = out / "sources.json"
    if meta.exists() and json.loads(meta.read_text()) != hashes:
        raise SystemExit("Source files changed since this run started; refusing to mix code")
    meta.write_text(json.dumps(hashes, indent=1) + "\n")
    cells = [
        (f, xi_i, n) for n in B.NS for f in range(len(B.FAMILIES)) for xi_i in range(len(B.XIS))
    ]
    if a.cells:
        cells = [cells[int(i)] for i in a.cells.split(",")]
    cal = {}
    for n in sorted({n for _, _, n in cells}):
        cal[n] = B.Calibration(B.grid(n))
        (out / f"calibration_n{n}.json").write_text(
            json.dumps(
                {
                    "ks": cal[n].ks,
                    "pareto": cal[n].pareto,
                    "union": cal[n].union,
                    "gp_c": cal[n].gp_c.tolist(),
                    "gp_xi_table": B.GP_XI_TABLE,
                },
                indent=1,
            )
            + "\n"
        )
    for f, xi_i, n in cells:
        fam, rho = B.FAMILIES[f]
        path = out / f"cell_{fam}_{rho}_{B.XIS[xi_i]}_{n}.json"
        if path.exists():
            continue
        t0 = time.time()
        jobs = [(a.seed, f, xi_i, n, r) for r in range(a.reps)]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cal,)) as pool:
            worlds = list(pool.map(_job, jobs, chunksize=max(1, a.reps // (4 * a.workers))))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "family": fam,
                    "rho_burr": rho,
                    "xi": B.XIS[xi_i],
                    "n": n,
                    "rho": B.true_rho(fam, rho, B.XIS[xi_i]),
                    "ks": B.grid(n),
                    "seed": a.seed,
                    "worlds": worlds,
                }
            )
            + "\n"
        )
        tmp.rename(path)
        print(
            f"{fam} {rho} xi={B.XIS[xi_i]} n={n}: {len(worlds)} worlds in {time.time() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
