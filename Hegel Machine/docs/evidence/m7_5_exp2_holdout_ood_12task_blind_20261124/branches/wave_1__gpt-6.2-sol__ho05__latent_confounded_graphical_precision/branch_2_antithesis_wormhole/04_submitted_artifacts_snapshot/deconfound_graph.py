"""Sparse conditional precision estimation with two Gaussian latent factors."""
import numpy as np
from scipy.linalg import cho_solve, eigh


def _profile(cov):
    """Profile out the optimal PSD rank-two covariance addition for each S."""
    d, u = eigh(cov, check_finite=False)
    root = (u * np.sqrt(d)) @ u.T
    eye = np.eye(len(cov))

    def evaluate(s):
        try:
            chol = np.linalg.cholesky(s)
        except np.linalg.LinAlgError:
            return np.inf, None
        # In coordinates whitened by S, the optimal covariance addition
        # lifts the two largest sample eigenvalues above the unit baseline.
        m = root @ s @ root
        values, vectors = eigh(m, subset_by_index=(len(cov) - 2, len(cov) - 1),
                               check_finite=False)
        values = np.maximum(values, 1.0)
        b = root @ vectors
        f = np.trace(m) - 2 * np.log(chol.diagonal()).sum()
        f -= np.sum(values - 1 - np.log(values))
        g = cov - (b * (1 - 1 / values)) @ b.T
        g -= cho_solve((chol, True), eye, check_finite=False)
        return f, (g + g.T) * 0.5

    return evaluate


def _fit(evaluate, s, penalty=0.0, support=None, iterations=120):
    """Proximal gradient with a positive-definite backtracking line search."""
    y = s.copy()
    step, momentum = 1.0, 1.0
    for _ in range(iterations):
        fy, g = evaluate(y)
        if g is None:
            y, momentum = s, 1.0
            fy, g = evaluate(y)
        while True:
            z = y - step * g
            if support is None:
                diag = z.diagonal().copy()
                z = np.sign(z) * np.maximum(np.abs(z) - step * penalty, 0.0)
                np.fill_diagonal(z, diag)
            else:
                z = np.where(support, z, 0.0)
            fz, _ = evaluate(z)
            delta = z - y
            bound = fy + np.sum(g * delta) + np.sum(delta * delta) / (2 * step)
            if fz <= bound + 1e-12:
                break
            step *= 0.5
        if np.max(np.abs(z - s)) < 2e-6:
            return z
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = z + ((momentum - 1) / next_momentum) * (z - s)
        s, momentum = z, next_momentum
        step = min(step * 1.12, 4.0)
    return s


def _residual_scale(cov):
    d, u = eigh(cov, check_finite=False)
    w = u[:, :-2] ** 2
    return np.sqrt((w @ (1 / d[:-2])) / w.sum(axis=1))


def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    """Return a hollow Boolean adjacency and an SPD precision on that support."""
    x = np.asarray(X_samples, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    sd = np.sqrt(np.mean(centered**2, axis=0))
    standardized = centered / sd
    corr = standardized.T @ standardized / n
    # Work in unit-invariant coordinates and estimate the scaling from the
    # nonfactor spectrum rather than the factor-inflated marginal variance.
    scale = _residual_scale(corr)
    for _ in range(5):
        scale *= _residual_scale(corr * np.outer(scale, scale)) ** 0.75
    cov = corr * np.outer(scale, scale)
    evaluate = _profile(cov)
    noise = 1 / np.sqrt(n)
    penalty = 0.8775 * noise
    pilot = _fit(evaluate, np.eye(p), penalty=penalty)
    def partial(s):
        return np.abs(s) / np.sqrt(np.outer(s.diagonal(), s.diagonal()))

    # Project normalized edge directions away from the fitted factor span.
    # Their retained lengths approximate information available for selection.
    d, u = eigh(pilot, check_finite=False)
    root = (u * np.sqrt(d)) @ u.T
    _, factors = eigh(root @ cov @ root, subset_by_index=(p - 2, p - 1),
                      check_finite=False)
    directions = root / np.sqrt(pilot.diagonal())
    retained = directions - factors @ (factors.T @ directions)
    gram = retained.T @ retained
    info = np.sqrt(np.clip(np.outer(gram.diagonal(), gram.diagonal()) + gram**2,
                           0.16, 1.0))
    weights = np.clip(1 - partial(pilot) / (12 * noise), 0.72, 1.0)
    pilot = _fit(evaluate, pilot, penalty=penalty * weights * info, iterations=100)
    adjacency = partial(pilot) >= 2.85 * noise / info**0.25
    np.fill_diagonal(adjacency, False)

    # Refit with the same latent model to undo the edge-selection shrinkage.
    support = adjacency | np.eye(p, dtype=bool)
    start = np.where(support, pilot, 0.0)
    if evaluate(start)[1] is None:
        start = np.diag(pilot.diagonal())
    units = scale / sd
    precision = _fit(evaluate, start, support=support) * np.outer(units, units)
    return {'adjacency': adjacency, 'sparse_precision': precision}
