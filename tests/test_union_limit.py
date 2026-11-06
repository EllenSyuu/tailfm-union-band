"""Checks of the finite-mesh union theory (preprint Sections 3–4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from theory_union_limit import beta, calibrate, coverage_width, g_paths, mesh  # noqa: E402


@pytest.mark.parametrize("rho", [-2.0, -1.0, -0.5, -0.3])
def test_beta_zero_at_truth_and_sign(rho):
    rp = np.array([-3.0, -2.5, -1.5, -0.7, -0.2])
    b = beta(rho, rp)
    assert beta(rho, np.array([rho]))[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(np.sign(b), np.sign(rp - rho))


def test_beta_ratio_to_hill_matches_lemma1_table():
    # r(ρ, ρ′) = β / (ρ/(1−ρ)); true ρ = −0.5 → −0.71, 0.40, 0.64 at ρ′ = −0.25, −1, −2
    r = beta(-0.5, np.array([-0.25, -1.0, -2.0])) / (-0.5 / 1.5)
    np.testing.assert_allclose(r, [-0.714, 0.4, 0.643], atol=2e-3)


def test_g_paths_pareto_centred_and_scaled():
    ks = np.array([64, 128, 256])
    rh = np.array([-1.0])
    g = g_paths(np.random.default_rng(1), 1500, ks, rh)
    assert abs(g.mean()) < 0.25  # small finite-k bias only
    np.testing.assert_allclose(g.std(axis=0)[:, 0], [2.0, 2.0, 2.0], rtol=0.15)  # s(−1) = 2


def test_union_covers_at_mesh_points_and_brackets():
    ks = np.array([64, 128, 256])
    t = ks / ks[-1]
    rh = np.sort(mesh(6))
    c = calibrate(g_paths(np.random.default_rng(2), 1500, ks, rh), rh)
    g = g_paths(np.random.default_rng(3), 1500, ks, rh)
    cov_mesh, _ = coverage_width(g, rh, c, t, float(rh[2]), 8.0)  # ρ ∈ M: ≥ 1 − α
    assert cov_mesh >= 0.87
    cov_far, _ = coverage_width(g, rh, c, t, -1.0, 40.0)  # ρ inside conv M: bracketing
    assert cov_far >= 0.9
    cov_out, _ = coverage_width(g, rh, c, t, -3.0, 40.0)  # ρ far below R: fails
    assert cov_out < 0.5


def test_limit_cov_closed_form_matches_pareto_simulation():
    from theory_union_limit import limit_cov

    ks = np.array([128, 256, 512])
    rh = np.array([-2.0, -0.5])
    S = limit_cov(ks / ks[-1], rh)
    assert np.linalg.eigvalsh(S).min() > -1e-9
    s2 = ((1 - rh) / np.abs(rh)) ** 2
    np.testing.assert_allclose(np.diag(S), np.tile(s2, 3), rtol=1e-10)
    g = g_paths(np.random.default_rng(4), 6000, ks, rh).reshape(6000, -1)
    np.testing.assert_allclose(np.cov(g.T), S, atol=0.25 * s2.max())


def test_uniform_endpoint_bound_holds_samplewise():
    """Theorem (uniform bound): on every draw, hull failure implies an endpoint exceedance."""
    from theory_union_limit import g_gauss

    ks = np.array([64, 128, 256, 512, 1024])
    t = ks / ks[-1]
    rh = np.sort(mesh(6))
    s = (1 - rh) / np.abs(rh)
    c = calibrate(g_gauss(np.random.default_rng(5), 4000, ks, rh), rh)
    g = g_gauss(np.random.default_rng(6), 4000, ks, rh)
    z = g / s
    for rho in (-2.0, -1.3, -0.8, -0.5):
        for lam in (-30.0, -3.0, 0.0, 3.0, 30.0):
            sh = (lam * t ** (0.5 - rho))[:, None] * beta(rho, rh)[None, :]
            lo = (g + sh - c * s).min(2)
            hi = (g + sh + c * s).max(2)
            fail = ~np.all((lo <= 0) & (hi >= 0), 1)
            if lam >= 0:
                ends = (z[:, :, 0].max(1) > c[0]) | (z[:, :, -1].min(1) < -c[-1])
            else:
                ends = (z[:, :, -1].max(1) > c[-1]) | (z[:, :, 0].min(1) < -c[0])
            assert not np.any(fail & ~ends)


def test_counterexample_two_sided_bracketing():
    """Proposition (counterexample): coverage tends to (1 - alpha)^2."""
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal((2, 200_000))
    c = 1.6448536269514722
    z1 = np.stack([x, -x, x, -x], 1)
    z2 = np.stack([y, -y, y, -y], 1)
    b = 1e3
    n1 = np.array([0, 0, -b, -b])
    n2 = np.array([b, b, 0, 0])
    lo = np.minimum(z1 + n1 - c, z2 + n2 - c)
    hi = np.maximum(z1 + n1 + c, z2 + n2 + c)
    cov = np.mean(np.all((lo <= 0) & (hi >= 0), 1))
    assert abs(cov - 0.81) < 0.005
