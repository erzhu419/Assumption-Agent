"""Complex upwind PDE inversion with exhaustive coupled support selection."""

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _basis_responses(problem, basis):
    m, nx, ny = basis.shape
    hx = float(problem.get('Lx', 1.0)) / (nx + 1)
    hy = float(problem.get('Ly', 1.0)) / (ny + 1)
    dx = float(problem['nu']) / hx**2
    dy = float(problem['nu']) / hy**2
    vx = np.asarray(problem['vx'], dtype=float) / hx
    vy = np.asarray(problem['vy'], dtype=float) / hy
    ids = np.arange(nx * ny).reshape(nx, ny)
    diagonal = (2 * (dx + dy) + np.abs(vx) + np.abs(vy)
                + np.asarray(problem['sigma'], dtype=float)
                + 1j * float(problem['omega']))

    # Each velocity belongs to the row (receiving stencil), not the neighbor.
    rows = (ids, ids[1:, :], ids[:-1, :], ids[:, 1:], ids[:, :-1])
    cols = (ids, ids[:-1, :], ids[1:, :], ids[:, :-1], ids[:, 1:])
    values = (diagonal, -dx - np.maximum(vx[1:, :], 0),
              -dx + np.minimum(vx[:-1, :], 0),
              -dy - np.maximum(vy[:, 1:], 0),
              -dy + np.minimum(vy[:, :-1], 0))
    matrix = coo_matrix(
        (np.concatenate([v.ravel() for v in values]),
         (np.concatenate([r.ravel() for r in rows]),
          np.concatenate([c.ravel() for c in cols]))),
        shape=(nx * ny, nx * ny), dtype=np.complex128).tocsc()
    return splu(matrix).solve(basis.reshape(m, -1).T)


def _select_support(g, y, lam2, lam0, pairs):
    m = g.shape[1]
    # One QR compresses every augmented ridge problem to at most M rows,
    # without squaring the condition number as normal equations would.
    augmented = np.vstack((g, np.sqrt(lam2) * np.eye(m)))
    q, r = np.linalg.qr(augmented, mode='reduced')
    target = q[:len(y)].conj().T @ y
    orthogonal = np.concatenate((y, np.zeros(m, dtype=np.complex128))) - q @ target
    constant = 0.5 * float(np.vdot(orthogonal, orthogonal).real)

    pair_masks = [((1 << int(u)) | (1 << int(v)), float(gamma))
                  for u, v, gamma in pairs]
    best_score = 0.5 * float(np.vdot(target, target).real)
    best_weights = np.zeros(m, dtype=np.complex128)
    for mask in range(1, 1 << m):
        penalty = lam0 * mask.bit_count() + sum(
            gamma for pair_mask, gamma in pair_masks if (mask & pair_mask) == pair_mask)
        if penalty >= best_score:
            continue
        support = [i for i in range(m) if mask & (1 << i)]
        columns = r[:, support]
        coeffs = lstsq(columns, target, lapack_driver='gelsy', check_finite=False)[0]
        if np.any(np.abs(coeffs) <= 1e-8):
            continue
        residual = columns @ coeffs - target
        score = 0.5 * float(np.vdot(residual, residual).real) + penalty
        if score < best_score:
            best_score = score
            best_weights.fill(0)
            best_weights[support] = coeffs
    return best_weights, float(best_score + constant)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    """Return complex coefficients, the penalized objective, and field magnitude."""
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    _, nx, ny = basis.shape
    responses = _basis_responses(problem, basis)
    sensors = np.asarray(problem['sensor_ij'], dtype=int)
    g = responses[sensors[:, 0] * ny + sensors[:, 1], :]
    y = (np.asarray(problem['y_obs_re'], dtype=float)
         + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    weights, objective = _select_support(
        g, y, float(problem['lam2']), float(problem['lam0']), problem['pair_penalties'])
    return {
        'weights_real': weights.real.copy(),
        'weights_imag': weights.imag.copy(),
        'objective': objective,
        'predicted_field_abs': np.abs((responses @ weights).reshape(nx, ny)),
    }
