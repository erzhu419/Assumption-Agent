"""Invert sparse thermistor measurements on a cell-centered Robin plate."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _thermal_operator(kappa, gamma):
    n = kappa.shape[0]
    h = 1.0 / n
    cells = np.arange(n * n).reshape(n, n)
    p = np.r_[cells[:-1, :].ravel(), cells[:, :-1].ravel()]
    q = np.r_[cells[1:, :].ravel(), cells[:, 1:].ravel()]
    k = kappa.ravel()
    g = 2 * k[p] * k[q] / (k[p] + k[q])
    robin = np.zeros_like(kappa)
    robin[0, :] += gamma * h
    robin[-1, :] += gamma * h
    robin[:, 0] += gamma * h
    robin[:, -1] += gamma * h
    diag = robin.ravel() + np.bincount(p, g, minlength=k.size) + np.bincount(q, g, minlength=k.size)
    ids = cells.ravel()
    return coo_matrix((np.r_[-g, -g, diag], (np.r_[p, q, ids], np.r_[q, p, ids])), shape=(k.size, k.size)).tocsc()


def _mad(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def _affine(pred, measured, weights):
    """Profile a nonnegative amplitude and a free common offset per column."""
    total = weights.sum(axis=0)
    pred_mean = (weights * pred).sum(axis=0) / total
    obs_mean = (weights * measured).sum(axis=0) / total
    centered = pred - pred_mean
    variance = (weights * centered**2).sum(axis=0)
    cov = (weights * centered * (measured - obs_mean)).sum(axis=0)
    gain = np.maximum(0., cov / np.maximum(variance, np.finfo(float).tiny))
    return gain, obs_mean - gain * pred_mean


def localize_thermal_source(measurement: dict) -> dict:
    """Fit source center and strength, allowing for gross errors at individual sensors."""
    n = int(measurement['N'])
    kappa = np.asarray(measurement['kappa'], dtype=float)
    sigma = float(measurement['sigma_m'])
    ij = np.asarray(measurement['sensor_ij'], dtype=int)
    obs = np.asarray(measurement['T_obs'], dtype=float).ravel()
    lu = splu(_thermal_operator(kappa, float(measurement['gamma_robin'])))
    idx = n * ij[:, 0] + ij[:, 1]
    # Reciprocity gives the entire source-to-thermistor map after one factorization.
    rhs = np.zeros((n*n, len(idx)), order='F')
    rhs[idx, np.arange(len(idx))] = 1.
    # Chunking avoids multithreaded BLAS overhead in SuperLU's many-RHS solve.
    greens = np.ascontiguousarray(np.concatenate([lu.solve(rhs[:, j:j+8]) for j in range(0, len(idx), 8)], axis=1).T)
    axis = (np.arange(n) + 0.5) / n
    coords = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    var = sigma**2
    magnitude = max(_mad(obs), np.finfo(float).eps * max(1., np.max(np.abs(obs))))
    floor = magnitude * 1e-9
    step_count = min(33, max(13, int(np.ceil(0.7 / sigma)) + 1))
    axis0 = np.linspace(0., 1., step_count)
    grid = np.stack(np.meshgrid(axis0, axis0, indexing='ij'), axis=-1).reshape(-1, 2)
    scale = 0.10 * magnitude

    def basis(xy):
        delta = coords - xy
        q = np.exp(-0.5 * np.sum(delta**2, axis=-1) / var) / n**2
        return delta, q

    roughness = []
    coefficients = []
    y = obs[:, None]
    for start in range(0, len(grid), 32):
        xy = grid[start:start+32]
        _, q = basis(xy[:, None, :])
        pred = np.einsum('ij,kj->ik', greens, q, optimize=False)
        weights = np.ones_like(pred)
        for _ in range(12):
            a, b = _affine(pred, y, weights)
            r = (pred * a + b - y) / scale
            weights = 1. / np.sqrt(1. + r*r)
        roughness.extend(np.sum(np.hypot(1., r) - 1., axis=0))
        coefficients.extend(zip(a, b))

    def prediction(xy, jacobian=False):
        delta, q = basis(xy)
        if jacobian:
            return np.einsum('ij,jk->ik', greens, q[:, None] * delta / var, optimize=False)
        return np.einsum('ij,j->i', greens, q, optimize=False)

    def fit(initial, loss, noise, keep):
        # Scale the linear variables by a representative temperature change.
        gain_scale = magnitude / max(np.std(prediction(initial[:2])), np.finfo(float).tiny)
        unit = np.array([1., 1., gain_scale, magnitude])

        def residual(z):
            return (z[2] * gain_scale * prediction(z[:2]) + z[3] * magnitude - obs)[keep]

        def jac(z):
            return np.column_stack((z[2] * gain_scale * prediction(z[:2], True),
                                    gain_scale * prediction(z[:2]), np.full(len(obs), magnitude)))[keep]

        result = least_squares(residual, initial / unit, jac=jac, loss=loss, f_scale=noise,
                               bounds=([0., 0., 0., -np.inf], [1., 1., np.inf, np.inf]),
                               gtol=1e-11, max_nfev=80)
        return result.x * unit, result.cost

    order = np.argsort(roughness)
    starts = []
    all_sensors = np.ones(len(obs), dtype=bool)
    prelim = []
    for i in order:
        if any(np.linalg.norm(grid[i] - p) < 0.16 for p in starts):
            continue
        starts.append(grid[i])
        guess = np.r_[grid[i], coefficients[i]]
        prelim.append(fit(guess, 'soft_l1', scale, all_sensors))
        if len(starts) == 3:
            break
    params = min(prelim, key=lambda result: result[1])[0]
    for _ in range(3):
        r = params[2] * prediction(params[:2]) + params[3] - obs
        noise = max(_mad(r), floor)
        params, _ = fit(params, 'cauchy', 2.5 * noise, all_sensors)
    def residuals(z):
        return z[2] * prediction(z[:2]) + z[3] - obs

    noise = max(_mad(residuals(params)), floor)
    cutoff = 3.3 * noise

    def refine(z):
        # Compare fits with the same bounded loss; a stuck sensor cannot win
        # simply because fitting it eliminates one extremely large residual.
        for _ in range(5):
            keep = np.abs(residuals(z)) < cutoff
            if keep.sum() < 5:
                return z, np.sum(np.minimum((residuals(z) / cutoff)**2, 1.))
            z, _ = fit(z, 'linear', 1., keep)
            if np.array_equal(np.abs(residuals(z)) < cutoff, keep):
                break
        return z, np.sum(np.minimum((residuals(z) / cutoff)**2, 1.))

    roots = [params]
    for z, _ in prelim:
        z, _ = fit(z, 'cauchy', 2.5 * noise, all_sensors)
        if all(np.linalg.norm(z[:2] - r[:2]) > .015 for r in roots):
            roots.append(z)
    winner = min((refine(z) for z in roots), key=lambda r: r[1])
    # A sensor near the heat peak can have a small residual precisely because
    # it has pulled the fit. Try deleting the highest-leverage inlier sensors.
    for _ in range(2):
        z, score = winner
        r = residuals(z)
        keep = np.abs(r) < cutoff
        indices = np.flatnonzero(keep)
        if len(indices) < 6:
            break
        design = np.column_stack((z[2] * prediction(z[:2], True), prediction(z[:2]), np.ones(len(obs))))[keep]
        leverage = np.sum(np.linalg.svd(design, full_matrices=False)[0]**2, axis=1)
        rank = leverage + .4 * np.minimum(np.abs(r[keep]) / cutoff, 1.)
        for j in indices[np.argsort(rank)[-8:]]:
            subset = keep.copy()
            subset[j] = False
            trial, _ = fit(z, 'linear', 1., subset)
            trial, trial_score = refine(trial)
            if trial_score < winner[1]:
                winner = trial, trial_score
        if winner[1] >= score - 1e-8:
            break
    # A trimmed fit uses the nominal contamination count to avoid rejecting
    # extra clean readings when a bad reading pulls a weakly observed fit.
    retained = len(obs) - int(round(.06 * len(obs)))
    if retained >= 5 and retained < len(obs):
        def trimmed(z):
            previous = None
            for _ in range(12):
                r = residuals(z)
                ids = np.argpartition(np.abs(r), retained - 1)[:retained]
                mask = np.zeros(len(obs), dtype=bool)
                mask[ids] = True
                if np.array_equal(mask, previous):
                    break
                previous = mask
                z, _ = fit(z, 'linear', 1., mask)
            score = np.partition(residuals(z)**2, retained - 1)[:retained].sum()
            return z, score

        candidates = [trimmed(z) for z in [winner[0], *roots, *(r[0] for r in prelim)]]
        best = min(candidates, key=lambda item: item[1])
        for _ in range(2):
            z, score = best
            r = residuals(z)
            ids = np.argpartition(np.abs(r), retained - 1)[:retained]
            design = np.column_stack((z[2] * prediction(z[:2], True), prediction(z[:2]), np.ones(len(obs))))[ids]
            leverage = np.sum(np.linalg.svd(design, full_matrices=False)[0]**2, axis=1)
            rank = leverage + .4 * np.minimum(np.abs(r[ids]) / cutoff, 1.)
            for j in ids[np.argsort(rank)[-8:]]:
                mask = np.zeros(len(obs), dtype=bool)
                mask[ids] = True
                mask[j] = False
                trial, _ = fit(z, 'linear', 1., mask)
                trial = trimmed(trial)
                if trial[1] < best[1]:
                    best = trial
            if best[1] >= score:
                break
        winner = refine(best[0])
    params = winner[0]
    return {'source_xy': params[:2].tolist(), 'amplitude': float(params[2]),
            'junction_offset': float(params[3])}
