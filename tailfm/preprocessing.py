"""Preserve magnitudes for xi; use tie-aware ranks only for copula inference."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

Array = NDArray[np.float64]


def validate_data(x: ArrayLike, *, n: int | None = None) -> Array:
    """Require finite positive numeric iid input (n,2), n>=32; never impute/drop rows.

    Positivity is a domain restriction, not evidence of regular variation or iid data.
    Both variables must vary. Ties are allowed with an explicit assumption warning.
    """
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise ValueError("Input must contain numeric columns, not strings or booleans")
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 32 or a.shape[1] != 2:
        raise ValueError("Expected numeric data of shape (n,2) with n>=32")
    if n is not None and len(a) != n:
        raise ValueError(f"Checkpoint requires exactly n={n} observations")
    if np.isnan(a).any():
        raise ValueError("Missing values are not supported; handle them explicitly before fitting")
    if not np.isfinite(a).all():
        raise ValueError("Infinite values are not supported")
    if np.any(a <= 0):
        raise ValueError("This milestone supports positive heavy-tailed marginals only")
    if np.any(np.ptp(a, axis=0) == 0):
        raise ValueError("Constant columns cannot identify tail behavior")
    if any(np.unique(a[:, j]).size < len(a) for j in range(2)):
        warnings.warn(
            "Tied observations: continuous-margin assumptions may fail", UserWarning, stacklevel=2
        )
    return a


def empirical_cdf(x: ArrayLike) -> Array:
    """In-sample average ranks/(n+1), shape (n,2), strictly inside (0,1).

    This removes marginal shape information. It is not an out-of-sample CDF fit.
    """
    a = validate_data(x)
    return rankdata(a, axis=0, method="average") / (len(a) + 1)


def tail_features(x: ArrayLike, k: int) -> Array:
    """Top-k sorted log(X/X_(n-k)), shape (2,k), invariant to positive scaling.

    Requires positive regularly varying upper tails. k trades bias against variance;
    no automatic threshold selection or invariance to arbitrary shifts is claimed.
    """
    a = validate_data(x)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 5 <= k < len(a):
        raise ValueError("k must be an integer with 5<=k<n")
    ordered = np.sort(a, axis=0)
    excess = np.log(ordered[-k:]) - np.log(ordered[-k - 1])
    if np.any(np.count_nonzero(excess > 0, axis=0) < 5):
        raise ValueError("Insufficient distinct positive tail exceedances (need at least five)")
    return excess.T


def prepare_inputs(x: ArrayLike, k: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return uniform-scale copula input (n,2) and marginal magnitude input (2,k).

    Computations use float64 before conversion to CPU model float32. Overflow is
    rejected; the logarithmic representation avoids raw extreme-value overflow.
    """
    copula, magnitude = empirical_cdf(x), tail_features(x, k)
    return copula.astype(np.float32), magnitude.astype(np.float32)
