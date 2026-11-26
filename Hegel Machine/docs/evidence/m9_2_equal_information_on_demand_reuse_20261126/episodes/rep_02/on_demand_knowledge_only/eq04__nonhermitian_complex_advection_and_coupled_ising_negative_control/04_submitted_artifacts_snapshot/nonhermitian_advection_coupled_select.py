"""Sparse complex PDE inversion with exhaustive pair-penalized ridge selection."""

from itertools import combinations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    m, nx, ny = basis.shape
    hx = problem['Lx'] / (nx + 1)
    hy = problem['Ly'] / (ny + 1)
    dx = problem['nu'] / hx**2
    dy = problem['nu'] / hy**2
    ux = np.asarray(problem['vx'], dtype=float) / hx
    uy = np.asarray(problem['vy'], dtype=float) / hy
    diag = (2 * (dx + dy) + np.abs(ux) + np.abs(uy)
            + np.asarray(problem['sigma'], dtype=float) + 1j * problem['omega'])
    ids = np.arange(nx * ny).reshape(nx, ny)
    rows = np.concatenate([ids.ravel(), ids[1:, :].ravel(), ids[:-1, :].ravel(),
                           ids[:, 1:].ravel(), ids[:, :-1].ravel()])
    cols = np.concatenate([ids.ravel(), ids[:-1, :].ravel(), ids[1:, :].ravel(),
                           ids[:, :-1].ravel(), ids[:, 1:].ravel()])
    values = np.concatenate([diag.ravel(),
                             (-dx - np.maximum(ux[1:, :], 0)).ravel(),
                             (-dx + np.minimum(ux[:-1, :], 0)).ravel(),
                             (-dy - np.maximum(uy[:, 1:], 0)).ravel(),
                             (-dy + np.minimum(uy[:, :-1], 0)).ravel()])
    operator = csc_matrix((values, (rows, cols)), shape=(nx * ny, nx * ny))
    return splu(operator).solve(basis.reshape(m, -1).T).reshape(nx, ny, m)


def solve_nonhermitian_coupled_inversion(problem: dict) -> dict:
    fields = _responses(problem)
    ij = np.asarray(problem['sensor_ij'], dtype=np.intp)
    g = fields[ij[:, 0], ij[:, 1], :]
    y = (np.asarray(problem['y_obs_re'], dtype=float)
         + 1j * np.asarray(problem['y_obs_im'], dtype=float))
    lam0, lam2 = problem['lam0'], problem['lam2']
    pairs = problem['pair_penalties']
    m = g.shape[1]
    # Keep the ridge rows explicit: forming G^H G can round away a
    # small positive ridge on linearly dependent sensor columns.
    design = np.vstack((g, np.sqrt(lam2) * np.eye(m)))
    target = np.concatenate((y, np.zeros(m, dtype=np.complex128)))
    best = 0.5 * float(np.vdot(y, y).real)
    weights = np.zeros(m, dtype=np.complex128)

    # Solve each support as augmented least squares, then score the
    # original objective, charging pairs only when both are active.
    for size in range(1, m + 1):
        indices = np.asarray(list(combinations(range(m), size)), dtype=np.intp)
        coeffs = np.array([np.linalg.lstsq(design[:, support], target, rcond=None)[0]
                           for support in indices])
        candidates = np.zeros((len(indices), m), dtype=np.complex128)
        candidates[np.arange(len(indices))[:, None], indices] = coeffs
        active = np.abs(candidates) > 1e-8
        residual = np.einsum('km,sm->sk', g, candidates) - y
        scores = (0.5 * np.sum(np.abs(residual)**2, axis=1)
                  + 0.5 * lam2 * np.sum(np.abs(coeffs)**2, axis=1)
                  + lam0 * active.sum(axis=1))
        for u, v, gamma in pairs:
            scores += gamma * (active[:, u] & active[:, v])
        i = int(np.argmin(scores))
        if scores[i] < best:
            best = float(scores[i])
            weights = candidates[i]

    return {'weights_real': weights.real, 'weights_imag': weights.imag,
            'objective': best, 'predicted_field_abs': np.abs(fields @ weights)}
