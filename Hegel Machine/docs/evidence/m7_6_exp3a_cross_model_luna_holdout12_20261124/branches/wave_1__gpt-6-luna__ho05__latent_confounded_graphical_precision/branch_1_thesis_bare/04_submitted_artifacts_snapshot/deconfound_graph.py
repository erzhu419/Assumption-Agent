"""Recover sparse precision support under a small number of latent factors."""
import numpy as np
from scipy.optimize import minimize


def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    """Estimate sparse precision S in Cov(X)^-1 = S - L, with L PSD low-rank.

    Fits a rank-two PSD correction by minimizing the off-diagonal L1 norm of
    the corrected precision. The returned support uses a conservative cutoff
    to distinguish sparse graph edges from residual factor-fit noise.
    """
    X = np.asarray(X_samples, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_samples must be a 2D array")
    n, p = X.shape
    if n < 2:
        raise ValueError("at least two samples are required")
    cov = np.cov(X, rowvar=False)
    # Small ridge protects the inverse from finite-sample near-singularity.
    scale = float(np.trace(cov) / p)
    cov = cov + np.eye(p) * max(scale, 1.0) * 1e-8
    precision = np.linalg.inv(cov)
    iu = np.triu_indices(p, 1)

    def objective(flat):
        U = flat.reshape(p, 2)
        corrected = precision + U @ U.T
        return np.abs(corrected[iu]).sum() + 1e-3 * np.square(U).sum()

    best = None
    for seed in (0, 1, 2):
        initial = np.random.default_rng(seed).normal(0.0, 0.3, size=(p, 2)).ravel()
        result = minimize(objective, initial, method="L-BFGS-B",
                          options={"maxiter": 4000, "ftol": 1e-12})
        if best is None or result.fun < best.fun:
            best = result
    U = best.x.reshape(p, 2)
    sparse = precision + U @ U.T
    sparse = (sparse + sparse.T) * 0.5
    adjacency = np.abs(sparse) >= 0.28
    np.fill_diagonal(adjacency, False)
    adjacency = adjacency | adjacency.T
    return {"adjacency": adjacency.astype(bool), "sparse_precision": sparse}
