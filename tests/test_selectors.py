import math

import numpy as np
import pytest

from tailfm import selectors as S
from tailfm.tails import draw, top_log_ratios


def _naive_samsee(x):
    """Direct transcription of eqs. (13)–(19) with explicit loops (reference)."""
    d = np.sort(x)[::-1]
    n = d.size

    def hill(k):
        return float(np.mean(np.log(d[:k] / d[k])))

    def hv(k):
        y = np.log(d[:k] / d[k])
        return float(np.mean(y**2) / (2 * hill(k)))

    def bbar(k, big_k):
        up = np.mean([hill(i) for i in range(k, big_k + 1)])
        return up - np.mean([hill(i) for i in range(1, big_k + 1)])

    def ad(big_k):
        return np.mean([(hv(k) + bbar(k, big_k) - hill(k)) ** 2 for k in range(1, big_k + 1)])

    lo, hi = max(4, math.ceil(0.1 * n)), n - 3
    best, k_star = np.inf, None
    for big_k in range(lo, hi + 1):
        val = abs(sum((ad(big_k + i) - ad(big_k)) / i for i in (-2, -1, 1, 2)))
        if val < best:
            best, k_star = val, big_k
    gj = 2 * hv(k_star) - hill(k_star)
    crit = {k: gj**2 / k + 4 * bbar(k, k_star) ** 2 for k in range(2, k_star)}
    return min(crit, key=crit.get), k_star


def _naive_ks(x, tail=0.15):
    d = np.sort(x)[::-1]
    big_t = int(math.floor(tail * d.size))
    best, kk = np.inf, None
    for k in range(2, big_t):
        h = float(np.mean(np.log(d[:k] / d[k])))
        dist = max(abs(d[j] - d[k - 1] * (k / j) ** h) for j in range(1, big_t + 1))
        if dist < best:
            best, kk = dist, k
    return kk


def test_hill_moments_match_log_ratios():
    x = draw("frechet", 120, 0.5, np.random.default_rng(np.random.SeedSequence([4990, 0]))).x
    h, m = S.hill_moments(x)
    for k in (1, 7, 60, 119):
        z = top_log_ratios(x, k)
        assert h[k - 1] == pytest.approx(z.mean(), rel=1e-12)
        assert m[k - 1] == pytest.approx(np.mean(z**2), rel=1e-9)


@pytest.mark.parametrize("family", ["frechet", "burr", "pareto_mix"])
def test_samsee_matches_direct_transcription(family):
    rng = np.random.default_rng(np.random.SeedSequence([4991, len(family)]))
    x = draw(family, 60, 0.6, rng, rho=-0.8).x
    out = S.samsee(x)
    k_ref, kstar_ref = _naive_samsee(x)
    assert (out["k"], out["K_star"]) == (k_ref, kstar_ref)
    assert out["xi"] == pytest.approx(top_log_ratios(x, out["k"]).mean())


@pytest.mark.parametrize("family", ["frechet", "abs_t", "gpd"])
def test_ks_distance_matches_direct_transcription(family):
    rng = np.random.default_rng(np.random.SeedSequence([4992, len(family)]))
    x = draw(family, 300, 0.4, rng).x
    out = S.ks_distance(x)
    assert out["k"] == _naive_ks(x) and out["T"] == 45
    assert out["xi"] == pytest.approx(top_log_ratios(x, out["k"]).mean())


def test_selectors_are_scale_invariant_and_validate_input():
    x = draw("burr", 400, 0.5, np.random.default_rng(np.random.SeedSequence([4993, 0])), -1.0).x
    for f in (S.samsee, S.ks_distance):
        a, b = f(x), f(9.0 * x)
        assert a["k"] == b["k"] and a["xi"] == pytest.approx(b["xi"])
        for bad in (x[:10], -x, x[:, None]):
            with pytest.raises(ValueError):
                f(bad)
    with pytest.raises(ValueError):
        S.ks_distance(x[:50], tail=0.05)


def test_selectors_recover_xi_on_large_frechet_sample():
    # Development sanity check (not a benchmark): ξ = 0.5, n = 5000.
    x = draw("frechet", 5000, 0.5, np.random.default_rng(np.random.SeedSequence([4994, 0]))).x
    assert abs(S.samsee(x)["xi"] - 0.5) < 0.05
    assert abs(S.ks_distance(x)["xi"] - 0.5) < 0.08
