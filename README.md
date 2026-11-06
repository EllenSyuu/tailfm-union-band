# tailfm-union-band

Code and stored outputs for the preprint

> Ziyi Zhou. *Selection-valid inference for the tail index under second-order bias without
> estimating the second-order parameter.* Preprint, 2026.

Release `v1.0.0` corresponds to version 1 of the preprint. Every number, table and figure
in the paper is generated from the files in `reports/` by `scripts/build_preprint_assets.py`.

## Contents

| Path | What it is |
|---|---|
| `tailfm/` | estimators (Hill, extended-Pareto), selection rules, the band construction and the simulation families |
| `scripts/` | benchmark, theory evaluations, post hoc analyses, Danish application, paper assets |
| `reports/phase3/` | stored outputs of every analysis (raw benchmark output included) |
| `paper/preprint/` | LaTeX source of the paper; `make_arxiv.sh` rebuilds it from `reports/` |
| `docs/phase-3-benchmark.md` | the benchmark protocol, written before the confirmatory run |
| `tests/` | unit tests, including checks of the closed-form covariance, the uniform bound and the counterexample |

## Install

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
pip install -e .
pytest -q                      # about 2 minutes
```

## Reproduce

Rebuild the paper from the stored outputs (needs `pdflatex`, `bibtex`, `latexmk`):

```bash
bash paper/preprint/make_arxiv.sh
```

Regenerate the stored outputs:

| Paper item | Command | Seeds | Status |
|---|---|---|---|
| Table 1, Section 6.2 | `python -I scripts/run_phase3_benchmark.py ...` then `scripts/summarise_phase3_benchmark.py` | 2033 | pre-registered |
| Figure 1, Remark 2 | `python -I scripts/theory_union_limit.py reports/phase3/theory --sims 20000 --kmax 1024` | 6460, 6461 | post hoc (theory) |
| Remark 2, Section 5 ($\pi_l$, $q_l$, $\varphi$, $U^+$, finite-$k$ skew) | `python -I scripts/theory_union_endpoint_bound.py reports/phase3/theory` | 6490–6493 | post hoc (theory) |
| Section 8, limitation 6 | `python -I scripts/theory_union_kappa.py reports/phase3/theory` | 6480, 6481 | post hoc (theory) |
| Figure 2, Section 6.3 | `python -I scripts/theory_union_predict_benchmark.py ...` | 6470, 6471 | post hoc |
| Section 6.4 | `python -I scripts/dev_phase3_rhohat_tuning.py ...` | 6450 | post hoc |
| Table 2, Section 6.5 | `python -I scripts/posthoc_comparators.py ...`, `scripts/posthoc_sasaki_wang_snooping.py` | regenerates the 2033 samples | post hoc |
| Table 3, Figure 3 | `python -I scripts/apply_danish.py ...` (needs `pip install -e '.[danish]'`) | none | frozen method |
| Appendix A identities | `python -I scripts/verify_lemmas_symbolic.py` | none | check |

Each script documents its arguments in its header. Approximate times on two CPU cores: theory
scripts about 10 minutes each, post hoc comparators about 20 minutes, the benchmark several
hours.

## Frozen benchmark code

The confirmatory benchmark was run once, on seed 2033, with code frozen before the run.
`reports/phase3/benchmark/sources.json` records the SHA-256 of each source file at run time;
the files `tailfm/phase3_bench.py`, `tailfm/phase3.py`, `tailfm/tail_estimators.py`,
`tailfm/selectors.py`, `tailfm/selectors_classical.py`, `tailfm/tails.py` and
`scripts/run_phase3_benchmark.py` in this repository match those hashes:

```bash
python - <<'EOF'
import hashlib, json
for f, h in json.load(open("reports/phase3/benchmark/sources.json")).items():
    print(f, hashlib.sha256(open(f, "rb").read()).hexdigest() == h)
EOF
```

Some docstrings in these frozen files refer to notes in the development repository; they are
left unchanged so that the hashes still match.

The protocol designated a plug-in band as the primary method and the union band as a robust
variant. The paper leads with the union band, which was decided after the confirmatory results
were seen; the paper states this, and all pre-registered results are reported unchanged.

## Data

The only real data set is the Danish fire insurance losses from the R package `evir`
(GPL ≥ 2). `scripts/apply_danish.py` downloads it at run time; it is not stored here.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).
