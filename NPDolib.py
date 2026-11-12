"""Projected and stochastic two-SVD NPDo methods for quadratic maximization.

Maximize tr(P.T A P) + 2 tr(P.T B), with P.T P = I and A positive
semidefinite. A may be a dense NumPy array or a SciPy sparse matrix.
Initial points are assumed feasible. Callbacks receive the block-update
count and the normalized KKT residual; both solvers return the final matrix.
"""

import numpy as np
from scipy import sparse


def _polar_factor(X):
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def _npdo_update(G, B):
    Y = _polar_factor(G)
    return Y @ _polar_factor(Y.T @ B)


def _scale(A, B):
    norm_A = sparse.linalg.norm(A, "fro") if sparse.issparse(A) else np.linalg.norm(A, "fro")
    value = float(norm_A + np.linalg.norm(B, "fro"))
    return value if value else 1.0  # The zero-data residual is defined as zero.


def _residual(P, H, scale):
    PH = P.T @ H
    return float(np.linalg.norm(H - P @ ((PH + PH.T) / 2), "fro") / scale)


def projected_npdo(A, B, Z, Y0, tol=1e-8, max_iter=1000, callback=None):
    """Algorithm 1: update Y with Y.T Y=I and Z.T Y=0 (Z.T Z=I).

    Each update uses polar(Pi H), then a right polar correction. An empty
    n-by-0 Z gives full NPDo. max_iter counts complete two-polar updates.
    """
    if max_iter < 0 or tol <= 0:
        raise ValueError("max_iter must be nonnegative and tol positive")
    Y = np.array(Y0, dtype=float, copy=True)
    scale = _scale(A, B)
    for i in range(max_iter + 1):
        H = 2 * (A @ Y) + 2 * B
        G = H - Z @ (Z.T @ H)
        YH = Y.T @ H
        residual = float(np.linalg.norm(G - Y @ ((YH + YH.T) / 2), "fro") / scale)
        if callback is not None:
            callback(i, residual)
        if residual <= tol or i == max_iter:
            break
        Y = _npdo_update(G, B)
    return Y


def stochastic_npdo(A, B, P0, d, tol=1e-8, max_iter=100000, rng=None,
                    callback=None):
    """Algorithm 2: sample a uniform d-column subset independently per update.

    max_iter counts sampled block updates, not sweeps. Global KKT checks
    occur every ceil(k/d) updates and at the final endpoint. This is the
    uncached sparse-problem path: no full AP/H cache is maintained.
    """
    P = np.array(P0, dtype=float, copy=True)
    k = P.shape[1]
    if not 2 <= d <= k or max_iter < 0 or tol <= 0:
        raise ValueError("require 2 <= d <= k, max_iter >= 0, and tol > 0")
    rng = np.random.default_rng(rng)
    scale = _scale(A, B)
    q = (k + d - 1) // d
    for i in range(max_iter + 1):
        if i % q == 0 or i == max_iter:
            residual = _residual(P, 2 * (A @ P) + 2 * B, scale)
            if callback is not None:
                callback(i, residual)
            if residual <= tol or i == max_iter:
                break
        I = np.sort(rng.choice(k, d, replace=False))
        Y = P[:, I].copy()
        B_I = B[:, I]
        H = 2 * (A @ Y) + 2 * B_I
        # Equivalent to (I-Z Z.T)H, where Z holds the unchanged columns.
        G = H - P @ (P.T @ H) + Y @ (Y.T @ H)
        P[:, I] = _npdo_update(G, B_I)
    return P
