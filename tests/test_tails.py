import numpy as np
import pytest

from tailfm.tails import (
    FAMILIES_2,
    FAMILIES_2B,
    TRAIN_RHO,
    TRAIN_XI,
    draw,
    draw_from_prior,
    second_order,
    survival,
    top_log_ratios,
)


@pytest.mark.parametrize("family", FAMILIES_2)
def test_tail_index_recovered_far_in_the_tail(family):
    rng = np.random.default_rng(np.random.SeedSequence([11, FAMILIES_2.index(family)]))
    sample = draw(family, 400_000, 0.4, rng, rho=-1.0)
    assert sample.x.min() > 0 and sample.xi == 0.4
    hill = top_log_ratios(sample.x, 1000).mean()
    assert abs(hill - 0.4) < 0.04


def test_second_order_parameters_and_bias_direction():
    assert second_order("frechet", 0.3) == -1
    assert second_order("abs_t", 0.25) == -0.5
    assert second_order("gpd", 0.7) == -0.7
    assert second_order("pareto", 0.5) == -np.inf
    with pytest.raises(ValueError):
        second_order("burr", 0.5)
    # Slow second order (rho near 0) biases Hill far more at a moderate threshold.
    rng = np.random.default_rng(3)
    fast = [top_log_ratios(draw("burr", 4000, 0.5, rng, -2.0).x, 400).mean() for _ in range(40)]
    slow = [top_log_ratios(draw("burr", 4000, 0.5, rng, -0.25).x, 400).mean() for _ in range(40)]
    assert abs(np.mean(slow) - 0.5) > 3 * abs(np.mean(fast) - 0.5)


def test_prior_draws_are_reproducible_and_in_box():
    a = draw_from_prior(64, np.random.default_rng(np.random.SeedSequence([5, 0])))
    b = draw_from_prior(64, np.random.default_rng(np.random.SeedSequence([5, 0])))
    np.testing.assert_array_equal(a.x, b.x)
    assert TRAIN_XI[0] <= a.xi <= TRAIN_XI[1] and TRAIN_RHO[0] <= a.rho <= TRAIN_RHO[1]
    g = draw_from_prior(64, np.random.default_rng(1), family="gpd")
    assert g.rho == -g.xi


def test_log_ratios_are_scale_invariant_and_validated():
    x = draw("pareto", 200, 0.5, np.random.default_rng(2)).x
    z = top_log_ratios(x, 20)
    np.testing.assert_allclose(top_log_ratios(7.5 * x, 20), z)
    assert z.shape == (20,) and np.all(np.diff(z) >= 0) and z[0] >= 0
    for bad, k in [(x, 0), (x, 200), (-x, 5), (x[:, None], 5)]:
        with pytest.raises(ValueError):
            top_log_ratios(bad, k)
    with pytest.raises(ValueError):
        draw("lognormal", 10, 0.5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        draw("pareto", 1, 0.5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        draw("pareto", 10, -0.5, np.random.default_rng(0))


@pytest.mark.parametrize("family", FAMILIES_2B)
def test_samples_follow_the_exact_survival_function(family):
    rng = np.random.default_rng(np.random.SeedSequence([4100, FAMILIES_2B.index(family)]))
    sample = draw(family, 200_000, 0.6, rng, rho=-0.8)
    qs = np.quantile(sample.x, [0.1, 0.5, 0.9, 0.99, 0.999])
    empirical = np.array([np.mean(sample.x > q) for q in qs])
    # Binomial standard error at p=0.1 with n=2e5 is about 7e-4; 5e-3 is > 6 SE.
    np.testing.assert_allclose(survival(family, qs, 0.6, rho=-0.8), empirical, atol=5e-3)
    lower = 0.0 if family in ("burr", "frechet", "abs_t") else 1.0
    assert survival(family, np.array([lower]), 0.6, rho=-0.8)[0] == 1.0
    assert sample.x.min() >= lower


def test_new_families_closed_forms_and_second_order():
    assert FAMILIES_2B[:5] == FAMILIES_2 and len(FAMILIES_2B) == 7
    # Pareto mixture at x=2, ξ=1: 0.5/2 + 0.5/4 = 0.375. Log-gamma at x=e, ξ=1: 2/e.
    assert survival("pareto_mix", 2.0, 1.0) == pytest.approx(0.375)
    assert survival("log_gamma", np.e, 1.0) == pytest.approx(2 / np.e)
    assert survival("log_gamma", 10.0, 0.5) == pytest.approx((1 + np.log(10) / 0.5) * 10**-2)
    # Pareto mixture: x^(1/ξ)F̄ = 0.5 + 0.5·x^(ρ/ξ) with ρ = −1 exactly.
    xi, x = 0.4, np.array([5.0, 50.0, 500.0])
    resid = x ** (1 / xi) * survival("pareto_mix", x, xi) - 0.5
    np.testing.assert_allclose(resid, 0.5 * x ** (-1 / xi))
    assert second_order("pareto_mix", xi) == -1.0 and second_order("log_gamma", xi) == 0.0
    # Log-gamma: the slowly varying factor makes Hill biased upwards at every threshold.
    rng = np.random.default_rng(np.random.SeedSequence([4101, 0]))
    hills = [top_log_ratios(draw("log_gamma", 20_000, 0.5, rng).x, 200).mean() for _ in range(20)]
    assert np.mean(hills) > 0.55
    with pytest.raises(ValueError):
        survival("lognormal", 2.0, 0.5)


def test_pareto_mixture_hill_bias_is_negative_as_delta_positive_implies():
    # F̄ = 0.5x^(−1/ξ)(1 + x^(−1/ξ)): the positive second-order term makes the local
    # log-spacings smaller than ξ, so Hill is biased downwards (opposite to Burr).
    rng = np.random.default_rng(np.random.SeedSequence([4102, 0]))
    hills = [top_log_ratios(draw("pareto_mix", 2000, 0.5, rng).x, 400).mean() for _ in range(40)]
    assert np.mean(hills) < 0.48
