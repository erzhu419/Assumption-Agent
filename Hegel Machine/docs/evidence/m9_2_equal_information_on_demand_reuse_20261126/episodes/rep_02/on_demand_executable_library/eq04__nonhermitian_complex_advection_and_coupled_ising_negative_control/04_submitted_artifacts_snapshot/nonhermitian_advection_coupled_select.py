"""Complex upwind forward model and exhaustive coupled ridge selection."""

from itertools import combinations

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem, basis):
    m, nx, ny = basis.shape
    hx = float(problem['Lx']) / (nx + 1)
    hy = float(problem['Ly']) / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    ux = np.asarray(problem['vx'], dtype=float) / hx
    uy = np.asarray(problem['vy'], dtype=float) / hy
    diag = (2 * (dx + dy) + np.abs(ux) + np.abs(uy)
            + np.asarray(problem['sigma'], dtype=float) + 1j * problem['omega'])

    ids = np.arange(nx * ny).reshape(nx, ny)
    rows = np.concatenate((ids.ravel(), ids[1:, :].ravel(), ids[:-1, :].ravel(),
                           ids[:, 1:].ravel(), ids[:, :-1].ravel()))
    cols = np.concatenate((ids.ravel(), ids[:-1, :].ravel(), ids[1:, :].ravel(),
                           ids[:, :-1].ravel(), ids[:, 1:].ravel()))
    values = np.concatenate((diag.ravel(),
                             (-dx - np.maximum(ux[1:, :], 0)).ravel(),
                             (-dx + np.minimum(ux[:-1, :], 0)).ravel(),
                             (-dy - np.maximum(uy[:, 1:], 0)).ravel(),
                             (-dy + np.minimum(uy[:, :-1], 0)).ravel()))
    operator = coo_matrix((values, (rows, cols)), shape=(ids.size, ids.size),
                          dtype=np.complex128).tocsc()
    rhs = basis.reshape(m, ids.size).T.astype(np.complex128)
    return splu(operator).solve(rhs)


def _select(design, observed, lam0, lam2, pairs):
    m = design.shape[1]
    best = np.zeros(m, dtype=np.complex128)
    # Compress the augmented ridge problem once. The discarded residual is
    # independent of the support and is included when evaluating the winner.
    augmented = np.vstack((design, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:observed.size].conj().T @ observed
    best_score = float(0.5 * np.vdot(target, target).real)
    for size in range(1, m + 1):
        for support in combinations(range(m), size):
            submatrix = r[:, support]
            weights = lstsq(submatrix, target, lapack_driver='gelsy',
                            check_finite=False)[0]
            residual = submatrix @ weights - target
            active = {i for i, w in zip(support, weights) if abs(w) > 1e-8}
            penalty = lam0 * len(active) + sum(gamma for u, v, gamma in pairs
                                               if u in active and v in active)
            score = float(0.5 * np.vdot(residual, residual).real + penalty)
            if score < best_score:
                best_score = score
                best.fill(0)
                best[list(support)] = weights

    residual = design @ best - observed
    active = np.abs(best) > 1e-8
    penalty = lam0 * np.count_nonzero(active) + sum(gamma for u, v, gamma in pairs
                                                   if active[u] and active[v])
    value = 0.5 * (np.vdot(residual, residual).real + lam2 * np.vdot(best, best).real)
    return best, float(value + penalty)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return the complex ridge minimizer selected across all supports."""
    basis = np.asarray(problem['basis'], dtype=float)
    _, nx, ny = basis.shape
    responses = _responses(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=np.intp)
    design = responses[sensors[:, 0] * ny + sensors[:, 1]]
    observed = (np.asarray(problem['y_obs_re'], dtype=float)
                + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    pairs = [(int(u), int(v), float(gamma)) for u, v, gamma in problem['pair_penalties']]
    weights, objective = _select(design, observed, float(problem['lam0']),
                                 float(problem['lam2']), pairs)
    return {'weights_real': weights.real.copy(),
            'weights_imag': weights.imag.copy(),
            'objective': objective,
            'predicted_field_abs': np.abs(responses @ weights).reshape(nx, ny)}
