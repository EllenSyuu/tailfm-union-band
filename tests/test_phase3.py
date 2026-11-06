import numpy as np
import pytest

from tailfm import phase3 as P3
from tailfm.tails import draw


def test_hierarchical_posterior_with_one_rho_equals_fixed_rho():
    x = draw("burr", 800, 0.5, np.random.default_rng(np.random.SeedSequence([6800, 0])), -1.0).x
    lx1, c1 = P3.epd_tail_posterior(x, 100, -1.0, nx=81, nd=21)
    lx2, c2 = P3.epd_tail_posterior_hier(x, 100, np.array([-1.0]), nx=81, nd=21)
    np.testing.assert_array_equal(lx1, lx2)
    np.testing.assert_allclose(c1, c2, rtol=1e-12)


def test_pareto_posterior_is_inverse_gamma():
    # Uniform prior on log ξ: 1/ξ ~ Gamma(k, rate S); median of ξ ≈ S/(k − 1/3).
    x = draw("pareto", 2000, 0.7, np.random.default_rng(np.random.SeedSequence([6800, 1]))).x
    k = 200
    lx, cdf = P3.pareto_tail_posterior(x, k)
    s = np.sort(np.log(x))
    total = (s[-k:] - s[-k - 1]).sum()
    assert P3.quantiles(lx, cdf, [0.5])[0] == pytest.approx(total / (k - 1 / 3), rel=2e-3)


def test_calibrated_level_between_pointwise_and_bonferroni():
    a_star, c = P3.calibrate_level([32, 64, 128, 256], -1.0, sims=400)
    assert 1.645 < c < 2.25 and 0.1 / 4 < a_star < 0.1
    a_par, c_par = P3.calibrate_level([32, 64, 128, 256], None, sims=400)
    assert 1.645 < c_par < 2.25


def test_gp_posterior_centres_on_truth_for_exact_gpd_excesses():
    # Exact Pareto tail: excesses over any threshold are GP with γ = ξ; average the posterior
    # median over samples and check it is within Monte Carlo error of ξ.
    meds = []
    for r in range(30):
        x = draw("pareto", 3000, 0.6, np.random.default_rng(np.random.SeedSequence([6801, r]))).x
        lg, cdf = P3.gp_tail_posterior(x, 300, ng=161, ns=81)
        meds.append(P3.quantiles(lg, cdf, [0.5])[0])
    # GP sd for γ = 0.6 at k = 300 is (1 + γ)/√k ≈ 0.092; s.e. of the mean ≈ 0.017.
    assert abs(np.mean(meds) - 0.6) < 0.05


def test_benchmark_grid_and_lepski():
    from tailfm import phase3_bench as B

    assert B.grid(2000) == (18, 37, 74, 149, 299)
    assert B.grid(5000) == (37, 74, 148, 297, 594)
    # Lepski stops before the first interval that misses an earlier band.
    iv = np.array(
        [[0.4, 0.5, 0.6, 0.3, 0.7], [0.45, 0.5, 0.55, 0.4, 0.6], [0.8, 0.9, 1.0, 0.7, 1.1]]
    )
    assert B.lepski(iv) == 1
    assert B.nearest((18, 37, 74), 50) == 1


def test_benchmark_gp_level_interpolates_table():
    from tailfm import phase3_bench as B

    cal = object.__new__(B.Calibration)
    cal.gp_c = np.linspace(2.0, 2.6, len(B.GP_XI_TABLE))
    a_lo, a_hi = cal.gp(0.01), cal.gp(10.0)
    assert a_lo == pytest.approx(2 * P3.stats.norm.sf(2.0))
    assert a_hi == pytest.approx(2 * P3.stats.norm.sf(2.6))
    assert a_hi < cal.gp(0.7) < a_lo
