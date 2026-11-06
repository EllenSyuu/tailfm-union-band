import math

import numpy as np
import pytest

from tailfm import selectors_classical as C
from tailfm.tails import draw, top_log_ratios


def _beta_dk(d, k):
    """β̂_{n,k} of Drees & Kaufmann p. 150: (k−1)⁻¹ Σ_{i=1}^{k−1} log(X_{n−i+1:n}/X_{n−k+1:n})."""
    return float(np.mean(np.log(d[: k - 1] / d[k - 1])))


def _naive_dk(x, rho=None):
    d = np.sort(x)[::-1]
    n = d.size

    def stat(k):
        return max(math.sqrt(i) * abs(_beta_dk(d, i) - _beta_dk(d, k)) for i in range(2, k + 1))

    def k_bar(r):
        while True:
            for k in range(2, n + 1):
                if stat(k) > r:
                    return k, r
            r *= 0.9

    b0 = _beta_dk(d, int(math.floor(2 * math.sqrt(n))))
    kb, r = k_bar(2.5 * b0 * n**0.25)
    kbx, _ = k_bar(r**0.7)
    if rho is None:
        lk = int(math.floor(0.6 * kbx))
        rho = math.log(stat(lk) / stat(kbx)) / math.log(0.6) - 0.5 if lk >= 2 else math.nan
        if not (np.isfinite(rho) and rho > 0):
            rho = 1.0
    k = math.floor(
        (2 * rho + 1) ** (-1 / rho)
        * (2 * b0**2 * rho) ** (1 / (2 * rho + 1))
        * (kbx / kb**0.7) ** (1 / 0.3)
    )
    if k < 2:
        k = int(math.floor(2 * math.sqrt(n)))
    return min(k, n)


def _naive_gh(x, crit=1.25):
    d = np.sort(x)[::-1]
    n = d.size
    y = [i * math.log(d[i - 1] / d[i]) for i in range(1, n)]

    def t(k):
        w = [np.sign(k - 2 * i + 1) * abs(k - 2 * i + 1) for i in range(1, k + 1)]
        h = np.mean(y[:k])
        return math.sqrt(3 / k**3) * sum(wi * yi for wi, yi in zip(w, y[:k], strict=True)) / h

    tt = {k: t(k) for k in range(1, n)}
    q = {}
    for k in range(2, n):
        lo = k // 2
        if k + lo <= n - 1:
            q[k] = math.sqrt(sum(tt[k + j] ** 2 for j in range(-lo, lo + 1)) / (2 * lo + 1))
    ks = sorted(q)
    for k in ks:
        if q[k] > crit:
            return k
    return ks[-1]


@pytest.mark.parametrize("family", ["frechet", "burr", "abs_t", "pareto_mix"])
def test_drees_kaufmann_matches_direct_transcription(family):
    rng = np.random.default_rng(np.random.SeedSequence([4995, len(family)]))
    x = draw(family, 120, 0.5, rng, rho=-0.7).x
    for rho in (None, 1.0):
        out = C.drees_kaufmann(x, rho=rho)
        k_dk = _naive_dk(x, rho)
        assert out["k"] == k_dk - 1
        assert out["xi"] == pytest.approx(top_log_ratios(x, out["k"]).mean())


@pytest.mark.parametrize("family", ["frechet", "burr", "gpd"])
def test_guillou_hall_matches_direct_transcription(family):
    rng = np.random.default_rng(np.random.SeedSequence([4996, len(family)]))
    x = draw(family, 150, 0.5, rng, rho=-1.0).x
    out = C.guillou_hall(x)
    assert out["k"] == _naive_gh(x)
    assert out["xi"] == pytest.approx(top_log_ratios(x, out["k"]).mean())


def test_guillou_hall_statistic_is_centred_under_exact_pareto():
    # Under exact Pareto the weighted spacings have mean zero and T_k is close to N(0, 1).
    rng = np.random.default_rng(np.random.SeedSequence([4997, 0]))
    vals = [C.guillou_hall_stats(draw("pareto", 400, 0.7, rng).x)[0][200] for _ in range(300)]
    assert abs(np.mean(vals)) < 0.15 and 0.8 < np.std(vals) < 1.2


def test_classical_selectors_scale_invariant_and_validated():
    x = draw("frechet", 600, 0.5, np.random.default_rng(np.random.SeedSequence([4998, 0]))).x
    for f in (C.drees_kaufmann, C.guillou_hall):
        a, b = f(x), f(7.0 * x)
        assert a["k"] == b["k"] and a["xi"] == pytest.approx(b["xi"])
        with pytest.raises(ValueError):
            f(x[:10])
    assert C.drees_kaufmann(x, rho=1.0)["rho"] == 1.0


def test_classical_selectors_sensible_on_average():
    # Development sanity check (not a benchmark): Fréchet ξ = 0.5, n = 2000, 20 samples.
    # GH as published (first exceedance of Q, which reproduces their Table 1) is noisy: on
    # these 20 samples k̂ ranged from 3 to 697 and the mean error was 0.09; DK's was 0.04.
    errs = {"dk": [], "gh": []}
    for s in range(20):
        rng = np.random.default_rng(np.random.SeedSequence([4999, s]))
        x = draw("frechet", 2000, 0.5, rng).x
        errs["dk"].append(abs(C.drees_kaufmann(x)["xi"] - 0.5))
        errs["gh"].append(abs(C.guillou_hall(x)["xi"] - 0.5))
    assert np.mean(errs["dk"]) < 0.06 and np.mean(errs["gh"]) < 0.12
