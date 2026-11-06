# Phase 3 confirmatory benchmark — pre-registration

Frozen 2026-11-04, before seed 2033 is used. The commit that adds this file pins the code;
`scripts/run_phase3_benchmark.py` records SHA-256 hashes of every source file in
`OUTDIR/sources.json` and refuses to resume if any of them changed. Spent test seeds
(2026, 2027, 2029, 2031) are not reused. Seed 2033 is used once, for this benchmark only.

## Question

Do selection-valid credible bands from the second-order (EPD) tail posterior keep coverage
of ξ after a data-driven choice of k, where first-order intervals do not, in the regime
where the bias is not negligible? Secondary: how do they compare with the GP posterior
(pivotal band) and with the frequentist twin (same estimator, Wald band)?

## Method (frozen in `tailfm/phase3_bench.py`)

* Grid: k_max = ⌊n^0.75⌋, k_j = ⌊k_max/2^j⌋, j = 0..4 (rule and status:
  `docs/theory/C4-gp-bias-quantiles-grid.md` §3).
* Primary method `epd_rhohat`: EPD tail posterior at every grid k with ρ̂ (FAGH at
  ⌊n^0.999⌋, capped at −0.05); band = central posterior interval at level 1 − α*, α* from
  the exact-Pareto calibration of the BJS estimator path at ρ̂ (α = 0.1).
* Robust method `epd_union`: union over ρ ∈ R = [−2, −0.5] (6 log-spaced points) of the
  known-ρ bands.
* Comparators: `pareto` (first-order posterior, Rényi-calibrated band); `gp` (Dombry,
  Padoan & Rizzelli posterior, pivotal band calibrated at the Hill estimate at k_max from a
  table over ξ ∈ {0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0}, interpolated in log ξ); `wald_epd`
  (BJS estimator ± c·se at ρ̂, same c as the primary band).
* Calibration seeds 6500 (Pareto/EPD) and 6600 (GP), computed once per n and stored in
  `OUTDIR/calibration_n{n}.json`; disjoint from the evaluation seed.
* Selection rules: SAMSEE, Guillou–Hall (each mapped to the nearest grid point in log k),
  Lepski-type stability on each method's own intervals and bands, largest grid k.

## Design

* Families: Burr ρ = −0.5, Burr ρ = −1, Fréchet (ρ = −1), |t| (ρ = −2ξ), and Burr ρ = −0.25.
  The last is **outside** the stated assumption ρ ≤ −0.5: reported, not in any decision rule.
* ξ ∈ {0.5, 1.0}; n ∈ {2000, 5000}; 20 cells, 16 in the assumption.
* 300 replications per cell. World r of cell (f, ξ_i, n) uses
  `SeedSequence([2033, 8, f, ξ_i, n], spawn_key=(r,))`.
* Command: `python scripts/run_phase3_benchmark.py OUTDIR --seed 2033 --reps 300`;
  summary: `python -I scripts/summarise_phase3_benchmark.py OUTDIR OUT.md OUT.json`.

## Metrics (per cell)

Simultaneous band coverage over the grid; coverage after selection of the 90% interval and
of the band at k̂ for each rule; pointwise coverage and bias by k; mean widths at k_max;
seconds per method. Monte Carlo s.e. of a coverage near 0.9 with 300 reps is ≈ 0.017.

## Decision rules (coded in `scripts/summarise_phase3_benchmark.py`; n_in = 16)

* **D1a** (primary claim): the `epd_rhohat` band after selection has coverage ≥ 0.85 for
  all four rules in ≥ n_in − 2 = 14 cells.
* **D1b** (robust claim): the `epd_union` band after selection has coverage ≥ 0.85 for all
  four rules in all 16 cells.
* **D2** (the problem is real): the first-order 90% interval after Lepski-type selection
  has coverage < 0.80 in ≥ 8 cells.
* **D3** (Bayes vs frequentist twin): the `epd_rhohat` simultaneous coverage is closer to
  0.9 than the Wald twin's in ≥ ⌈0.625 · 16⌉ = 10 cells.
* **D4** (reported only, no claim): number of cells where the `epd_rhohat` 90% interval at
  k_max is narrower than GP's with both simultaneous coverages ≥ 0.85.
* **Main claims supported** iff D1a ∧ D1b ∧ D2.

## What each outcome means (fast stop)

* Main claims supported: the manuscript reports them as confirmed, with D3/D4 as stated.
* D1a fails but D1b holds: the paper's primary method becomes the union band; the ρ̂
  version is reported as failing where it fails. No re-run on a new seed to rescue D1a.
* D1b fails: the selection-validity claim is not supported by this benchmark; the paper
  does not go to a top venue on this result. Stop and reassess before any further work.
* D2 fails: the problem is not shown at these n; the paper cannot claim a practical need.
* Every number is reported whatever the outcome, including the out-of-assumption cells.

## Development runs preceding the freeze

Dev seeds 6300 (smoke, 4–6 reps) and 6301 (pilot, 40 reps per cell) on the frozen code,
to check run time and that the pipeline completes. The decision thresholds above were set
before the pilot was run and not changed after it.
