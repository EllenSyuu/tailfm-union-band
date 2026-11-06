import numpy as np
import pytest

from tailfm.tail_estimators import (
    bayes_burr,
    bayes_epd_path,
    bayes_hill_path,
    epd_bayes_path,
    epd_loglik,
    epd_path,
    hill_path,
    mvrb_path,
    second_order_estimates,
)
from tailfm.tails import draw, draw_from_prior, top_log_ratios


def test_hill_path_matches_definition():
    x = draw("burr", 300, 0.4, np.random.default_rng(1), -1.0).x
    s = np.sort(x)
    for k, h in zip([5, 40, 299], hill_path(x, [5, 40, 299]), strict=True):
        assert h == pytest.approx(np.mean(np.log(s[-k:])) - np.log(s[-k - 1]))
    np.testing.assert_allclose(hill_path(3 * x, [40]), hill_path(x, [40]))
    for bad in [[1], [300], []]:
        with pytest.raises(ValueError):
            hill_path(x, bad)
    with pytest.raises(ValueError):
        hill_path(x[:10], [3])


def test_second_order_scale_matches_burr_and_mvrb_reduces_bias():
    rng = np.random.default_rng(np.random.SeedSequence([4, 4]))
    estimates = [second_order_estimates(draw("burr", 50_000, 0.5, rng, -1.0).x) for _ in range(4)]
    assert all(r < 0 for r, _ in estimates)
    assert abs(np.mean([b for _, b in estimates]) - 1.0) < 0.1  # Burr has beta = 1
    hill_bias, mvrb_bias, epd_bias = [], [], []
    for _ in range(60):
        x = draw("burr", 2000, 0.5, rng, -0.5).x
        hill_bias.append(hill_path(x, [200])[0] - 0.5)
        mvrb_bias.append(mvrb_path(x, [200])[0] - 0.5)
        epd_bias.append(epd_path(x, [200])[0] - 0.5)
    assert abs(np.mean(mvrb_bias)) < 0.7 * abs(np.mean(hill_bias))
    assert abs(np.mean(epd_bias)) < 0.3 * abs(np.mean(hill_bias))
    with pytest.raises(ValueError):
        second_order_estimates(np.arange(1.0, 40.0), k1=10)
    with pytest.raises(ValueError):
        epd_path(np.arange(1.0, 40.0), [10], rho=0.5)


def test_bayes_controls_stay_in_prior_and_bayes_burr_beats_hill_in_model():
    rng = np.random.default_rng(np.random.SeedSequence([9, 1]))
    x = draw("burr", 512, 0.5, rng, -1.0).x
    for path in (bayes_hill_path(x, [16, 64]), epd_bayes_path(x, [16, 64], grid=(31, 5, 9))):
        assert np.all((path >= 0.05) & (path <= 1.0))
    hill_error, bayes_error = [], []
    for _ in range(40):
        s = draw_from_prior(512, rng)
        hill_error.append(abs(hill_path(s.x, [32])[0] - s.xi))
        bayes_error.append(abs(bayes_burr(top_log_ratios(s.x, 128), 512, grid=(61, 9, 60)) - s.xi))
    assert np.mean(bayes_error) < 0.75 * np.mean(hill_error)
    with pytest.raises(ValueError):
        bayes_burr(np.array([0.1]), 512)
    with pytest.raises(ValueError):
        bayes_burr(np.array([-0.1, 0.2]), 512)


def test_epd_loglik_is_the_density_of_the_epd_survival_function():
    # Eq. (8) of Maribe et al. must be log of −dF̄/dy for F̄(y) = {y(1+δ−δy^τ)}^(−1/ξ).
    xi, d, tau = 0.6, 0.4, -1.3
    y = np.array([1.2, 2.0, 5.0, 30.0])
    h = 1e-6

    def surv(t):
        return (t * (1 + d - d * t**tau)) ** (-1 / xi)

    numeric = np.log((surv(y - h) - surv(y + h)) / (2 * h)).sum()
    assert epd_loglik(np.log(y), xi, d, tau) == pytest.approx(numeric, rel=1e-7)
    # δ = 0 reduces to the Pareto log-likelihood −k log ξ − (1/ξ+1)Σ log y.
    ly = np.log(y)
    assert epd_loglik(ly, xi, 0.0, tau) == pytest.approx(-4 * np.log(xi) - (1 / xi + 1) * ly.sum())
    assert epd_loglik(ly, -0.1, 0.0, tau) == -np.inf
    assert epd_loglik(ly, xi, -1.5, tau) == -np.inf  # 1 + δ(1 − y^τ) < 0


def test_bayes_epd_reduces_to_the_mdi_hill_mode_when_the_delta_prior_degenerates():
    # σ² → 0 forces δ = 0; the posterior mode of ξ under k·H/ξ-likelihood and the MDI
    # prior e^(−ξ)/ξ solves ξ² + (k+1)ξ − kH = 0.
    x = draw("burr", 500, 0.5, np.random.default_rng(4200), -1.0).x
    ks = np.array([20, 100])
    hills = hill_path(x, ks)
    exact = (-(ks + 1) + np.sqrt((ks + 1) ** 2 + 4 * ks * hills)) / 2
    np.testing.assert_allclose(bayes_epd_path(x, ks, sigma2=1e-10), exact, rtol=1e-6)
    with pytest.raises(ValueError):
        bayes_epd_path(x, ks, rho=0.2)


def test_bayes_epd_bias_reduction_on_burr_matches_the_paper_qualitatively():
    # Maribe et al., Section 4: Burr ξ=0.75, ρ=−0.75, n=500. At large k the Bayesian
    # estimator removes most of Hill's bias and has smaller MSE than both Hill and
    # EPD-ML. Small development replicate (seed ≥ 4000).
    errs = {"hill": [], "ml": [], "bayes": []}
    for r in range(25):
        rng = np.random.default_rng(np.random.SeedSequence([4301, r]))
        x = draw("burr", 500, 0.75, rng, -0.75).x
        rho = min(-0.5, second_order_estimates(x)[0])
        errs["hill"].append(hill_path(x, [300])[0] - 0.75)
        errs["ml"].append(epd_path(x, [300], rho=rho)[0] - 0.75)
        errs["bayes"].append(bayes_epd_path(x, [300], rho=rho)[0] - 0.75)
    mse = {m: np.mean(np.square(v)) for m, v in errs.items()}
    assert abs(np.mean(errs["bayes"])) < 0.5 * abs(np.mean(errs["hill"]))
    assert mse["bayes"] < mse["hill"] and mse["bayes"] < mse["ml"]
