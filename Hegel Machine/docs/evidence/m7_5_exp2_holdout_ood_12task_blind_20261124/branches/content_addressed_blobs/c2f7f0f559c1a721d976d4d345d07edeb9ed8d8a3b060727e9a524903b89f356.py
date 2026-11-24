"""Sparse precision estimation with two unobserved Gaussian factors."""

import numpy as np
from scipy.linalg import cho_solve, cholesky, eigh, solve_triangular
from scipy.optimize import minimize


def _profile(cov, precision, identity):
    """Profile the likelihood of cov(X) = precision^-1 + B B.T over B."""
    chol = cholesky(precision, lower=True, check_finite=False)
    conditional_cov = cho_solve((chol, True), identity, check_finite=False)
    p = len(cov)
    values, vectors = eigh(chol.T @ cov @ chol,
                           subset_by_index=(p - 2, p - 1), check_finite=False)
    # In whitened coordinates, the fitted factor variances are the two
    # leading sample eigenvalues in excess of the unit residual variance.
    excess = np.maximum(values - 1.0, 0.0)
    loadings = solve_triangular(chol.T, vectors, check_finite=False)
    residual = cov - (loadings * excess) @ loadings.T
    loss = (np.sum(precision * residual) - 2 * np.log(chol.diagonal()).sum()
            + np.log1p(excess).sum())
    return loss, residual - conditional_cov, residual


def _fit(cov, initial, edges, penalty):
    """Optimize the profiled likelihood on the supplied symmetric support."""
    p = len(cov)
    identity = np.eye(p)
    row, col = edges
    smooth = 0.006  # smoothing width for the off-diagonal l1 penalty

    def unpack(params):
        precision = np.diag(params[:p])
        precision[row, col] = precision[col, row] = params[p:]
        return precision

    def objective(params):
        try:
            loss, grad, _ = _profile(cov, unpack(params), identity)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(params)
        weights = params[p:]
        magnitude = np.hypot(weights, smooth)
        loss += 2 * penalty * np.sum(magnitude - smooth)
        return loss, np.r_[grad.diagonal(),
                          2 * (grad[row, col] + penalty * weights / magnitude)]

    start = np.r_[initial.diagonal(), initial[row, col]]
    # Masking an SPD matrix can lose positive definiteness. Shift the
    # diagonal of the starting point, leaving the selected edges unchanged.
    minimum = np.linalg.eigvalsh(unpack(start))[0]
    start[:p] += max(0.0, 0.075 - minimum)
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options={'maxiter': 160, 'ftol': 2e-11, 'gtol': 2e-6})
    return unpack(result.x)


def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    """Return the estimated graph and its support-constrained precision."""
    x = np.asarray(X_samples, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (n - 1)
    edges = np.triu_indices(p, 1)

    # Normalize the overall units without rescaling each variable by its
    # observed (factor-inflated) variance.
    noise = 1.0 / np.median(np.linalg.inv(cov).diagonal())
    cov /= noise
    values, vectors = eigh(cov, subset_by_index=(p - 2, p - 1))
    initial = np.linalg.inv(cov - (vectors * np.maximum(values - 1, 0)) @ vectors.T)
    fit = _fit(cov, initial, edges, penalty=0.041 * np.sqrt(900 / n))

    # Screen using unshrunk partial correlations in the factor-adjusted
    # sample, then refit without a penalty on the selected support.
    _, _, residual = _profile(cov, fit, np.eye(p))
    unshrunk = np.linalg.inv(residual)
    row, col = edges
    scores = np.abs(unshrunk[row, col]) / np.sqrt(unshrunk[row, row] * unshrunk[col, col])
    keep = scores > 3.55 / np.sqrt(n - p)
    selected = row[keep], col[keep]
    adjacency = np.zeros((p, p), dtype=bool)
    adjacency[selected] = True
    adjacency |= adjacency.T
    precision = _fit(cov, unshrunk, selected, penalty=0.0) / noise
    return {'adjacency': adjacency, 'sparse_precision': precision}
