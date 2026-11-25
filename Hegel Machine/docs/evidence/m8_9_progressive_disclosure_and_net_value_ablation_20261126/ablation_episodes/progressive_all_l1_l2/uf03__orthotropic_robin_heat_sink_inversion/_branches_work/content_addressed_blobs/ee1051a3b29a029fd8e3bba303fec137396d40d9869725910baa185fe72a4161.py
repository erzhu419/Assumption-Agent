"""Recover a Gaussian heat source using a conservative finite-volume model."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _sensor_green(kappa, gamma, sensors):
    """Rows of the inverse operator at the sensor cells, including cell area."""
    n = kappa.shape[0]
    cells = np.arange(n * n).reshape(n, n)
    a = np.r_[cells[:-1, :].ravel(), cells[:, :-1].ravel()]
    b = np.r_[cells[1:, :].ravel(), cells[:, 1:].ravel()]
    k = kappa.ravel()
    g = 2 * k[a] * k[b] / (k[a] + k[b])
    diag = np.bincount(np.r_[a, b], weights=np.r_[g, g], minlength=n*n)
    faces = np.zeros((n, n))
    faces[0, :] += 1
    faces[-1, :] += 1
    faces[:, 0] += 1
    faces[:, -1] += 1
    diag += (gamma / n) * faces.ravel()
    ids = cells.ravel()
    operator = coo_matrix((np.r_[diag, -g, -g], (np.r_[ids, a, b],
                           np.r_[ids, b, a])), shape=(n*n, n*n)).tocsc()
    factor = splu(operator)
    at = cells[sensors[:, 0], sensors[:, 1]]
    unique, inverse = np.unique(at, return_inverse=True)
    rhs = np.zeros((n*n, unique.size))
    rhs[unique, np.arange(unique.size)] = 1
    return (factor.solve(rhs).T[inverse] / (n*n)).reshape(-1, n, n)


def _grid_seeds(green, axis, sigma, obs, grid, keep):
    """Profile the two linear parameters on the position grid robustly."""
    basis = np.exp(-0.5 * ((grid[:, None] - axis) / sigma)**2)
    first = np.einsum('mij,pi->pmj', green, basis)
    field = np.einsum('pmj,qj->pqm', first, basis).reshape(-1, obs.size)
    weights = np.ones_like(field)
    tiny = np.finfo(float).tiny
    for step in range(8):
        wsum = weights.sum(axis=1)
        fbar = (weights * field).sum(axis=1) / wsum
        ybar = (weights @ obs) / wsum
        fc = field - fbar[:, None]
        yc = obs - ybar[:, None]
        amp = np.maximum(0, (weights * fc * yc).sum(axis=1) /
                         np.maximum((weights * fc**2).sum(axis=1), tiny))
        offset = ybar - amp * fbar
        resid = np.abs(yc - amp[:, None] * fc)
        if step < 4:
            scale = np.maximum(np.median(resid, axis=1), 1e-15)
            weights = np.minimum(1, 1.7 * scale[:, None] / np.maximum(resid, tiny))
        else:
            good = np.argpartition(resid, keep - 1, axis=1)[:, :keep]
            weights.fill(0)
            np.put_along_axis(weights, good, 1, axis=1)
    score = np.sum(np.partition(resid, keep - 1, axis=1)[:, :keep]**2, axis=1)
    seeds = []
    for idx in np.argsort(score):
        if amp[idx] <= 0:
            continue
        i, j = divmod(idx, grid.size)
        xy = np.array([grid[i], grid[j]])
        if all(np.linalg.norm(xy - seed[:2]) >= 0.13 for seed in seeds):
            seeds.append(np.r_[xy, amp[idx], offset[idx]])
        if len(seeds) == 5:
            break
    if not seeds:
        idx = np.argmin(score)
        i, j = divmod(idx, grid.size)
        seeds.append(np.array([grid[i], grid[j], 1., offset[idx]]))
    return seeds


def localize_thermal_source(measurement: dict) -> dict:
    """Estimate source position, positive amplitude and thermistor offset."""
    kappa = np.asarray(measurement['kappa'], dtype=float)
    sensors = np.asarray(measurement['sensor_ij'], dtype=int)
    obs = np.asarray(measurement['T_obs'], dtype=float).reshape(-1)
    n = int(measurement['N'])
    axis = (np.arange(n) + 0.5) / n
    sigma = float(measurement['sigma_m'])
    green = _sensor_green(kappa, float(measurement['gamma_robin']), sensors)
    keep = min(obs.size, max(4, int(np.ceil(0.88 * obs.size))))
    grid = np.linspace(0., 1., 23)
    seeds = _grid_seeds(green, axis, sigma, obs, grid, keep)
    green = green.reshape(obs.size, -1)
    sigma2 = sigma**2

    def model(p):
        dx, dy = axis - p[0], axis - p[1]
        q = np.exp(-0.5 * (dx[:, None]**2 + dy**2) / sigma2)
        loads = np.array([q, q * dx[:, None], q * dy]).reshape(3, -1)
        f, fx, fy = np.einsum('mk,lk->lm', green, loads)
        r = p[2] * f + p[3] - obs
        jac = np.column_stack((p[2] / sigma2 * fx, p[2] / sigma2 * fy,
                               f, np.ones(obs.size)))
        return r, jac

    def trimmed(p):
        r, jac = model(p)
        good = np.argsort(np.abs(r))[:keep]
        return r[good], jac[good]

    bounds = ([0., 0., 0., -np.inf], [1., 1., np.inf, np.inf])
    solutions = [least_squares(lambda p: trimmed(p)[0], seed,
                               jac=lambda p: trimmed(p)[1], bounds=bounds,
                               x_scale='jac', max_nfev=65) for seed in seeds]
    p = min(solutions, key=lambda fit: fit.cost).x

    # Use the trimmed fit to identify gross faults, then restore the regular
    # observations that were trimmed only for robust source localization.
    for _ in range(2):
        r, _ = model(p)
        center = np.median(r)
        scale = 1.4826 * np.median(np.abs(r - center))
        good = np.abs(r - center) <= max(3.5 * scale, 1e-12)
        p = least_squares(lambda p: model(p)[0][good], p,
                          jac=lambda p: model(p)[1][good], bounds=bounds,
                          x_scale='jac', max_nfev=40).x
    return {'source_xy': [float(p[0]), float(p[1])],
            'amplitude': float(p[2]), 'junction_offset': float(p[3])}
