"""Conservative cell-centered Robin heat-source inversion."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _sensor_green(kappa, robin, sensors):
    """Return Green functions indexed by sensor, plate x, and plate y."""
    n = kappa.shape[0]
    cells = np.arange(n*n).reshape(n, n)
    a = np.r_[cells[:-1, :].ravel(), cells[:, :-1].ravel()]
    b = np.r_[cells[1:, :].ravel(), cells[:, 1:].ravel()]
    ka, kb = kappa.ravel()[a], kappa.ravel()[b]
    conductance = 2 * ka * kb / (ka + kb)
    wall = np.zeros_like(kappa)
    wall[0, :] += 1
    wall[-1, :] += 1
    wall[:, 0] += 1
    wall[:, -1] += 1
    d = cells.ravel()
    matrix = coo_matrix(
        (np.r_[conductance, conductance, -conductance, -conductance,
               wall.ravel() * (robin/n)],
         (np.r_[a, b, a, b, d], np.r_[a, b, b, a, d])),
        shape=(n*n, n*n),
    ).tocsc()
    lu = splu(matrix)
    rows = cells[tuple(sensors.T)]
    green = np.empty((len(rows), n*n))
    for start in range(0, len(rows), 16):
        ix = rows[start:start+16]
        rhs = np.zeros((n*n, len(ix)), order='F')
        rhs[ix, np.arange(len(ix))] = 1
        green[start:start+len(ix)] = lu.solve(rhs).T / (n*n)
    return green.reshape(-1, n, n)


def _mad(residual, axis=-1):
    mid = np.median(residual, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(residual - mid), axis=axis, keepdims=True)


def _affine(x, y, w):
    total = w.sum(axis=1, keepdims=True)
    xm = (w*x).sum(axis=1, keepdims=True) / total
    ym = (w*y).sum(axis=1, keepdims=True) / total
    xc = x - xm
    xx = (w*xc*xc).sum(axis=1, keepdims=True)
    gain = np.maximum(0, (w*xc*(y-ym)).sum(axis=1, keepdims=True) /
                      np.maximum(xx, np.finfo(float).tiny))
    return gain, ym - gain*xm


def localize_thermal_source(measurement: dict) -> dict:
    """Estimate the center, positive amplitude and common junction offset."""
    n = int(measurement['N'])
    y = np.asarray(measurement['T_obs'], dtype=float)
    sigma = float(measurement['sigma_m'])
    green = _sensor_green(np.asarray(measurement['kappa'], dtype=float),
                          float(measurement['gamma_robin']),
                          np.asarray(measurement['sensor_ij'], dtype=int))
    cell = (np.arange(n)+0.5)/n
    s2 = sigma*sigma
    signal = max(np.ptp(y), 1e-12)
    floor = max(1e-10 * signal, 1e-13)
    # Use a common temperature scale to condition the amplitude coordinate.
    offset_ref = np.median(y)
    y = y - offset_ref
    green_scale = np.median(green.sum(axis=(1, 2))) * min(1, 2*np.pi*s2)
    green /= green_scale

    count = int(np.clip(np.ceil(1.5/sigma)+1, 15, 35))
    grid = np.linspace(0., 1., count)
    g = np.exp(-0.5*(grid[:, None]-cell)**2/s2)
    first = np.einsum('pi,mij->pmj', g, green)
    responses = np.einsum('pmj,qj->pqm', first, g).reshape(-1, len(y))
    w = np.ones_like(responses)
    for step in range(7):
        gain, offset = _affine(responses, y, w)
        r = gain*responses + offset - y
        scale = np.maximum(_mad(r), floor)
        w = 1 / np.sqrt(1 + (r / ((1.5 if step < 4 else 1.)*scale))**2)
    gain, offset = _affine(responses, y, w)
    keep = len(y) - max(1, int(np.ceil(0.08*len(y))))

    def trimmed(r):
        return np.partition(r*r, keep-1, axis=-1)[..., :keep].sum(axis=-1)

    scores = trimmed(gain*responses + offset - y)
    order = np.argsort(scores)
    starts = []
    for k in order:
        xy = np.array([grid[k//count], grid[k%count]])
        if all(np.linalg.norm(xy - old[:2]) >= max(0.09, sigma) for old in starts):
            starts.append(np.r_[xy, max(gain[k, 0], 0.01*signal), offset[k, 0]])
            if len(starts) == 5:
                break

    def evaluate(p):
        dx, dy = cell-p[0], cell-p[1]
        gx, gy = np.exp(-0.5*dx*dx/s2), np.exp(-0.5*dy*dy/s2)
        along_y = np.einsum('mij,j->mi', green, gy)
        v = np.einsum('mi,i->m', along_y, gx)
        vx = np.einsum('mi,i->m', along_y, gx*dx/s2)
        vy = np.einsum('mij,i,j->m', green, gx, gy*dy/s2)
        return p[2]*v + p[3] - y, np.column_stack((p[2]*vx, p[2]*vy, v, np.ones_like(v)))

    lower, upper = [0., 0., 0., -np.inf], [1., 1., np.inf, np.inf]
    param_scale = [0.2, 0.2, signal, signal]

    def fit(p, mask, loss, scale):
        return least_squares(lambda q: evaluate(q)[0][mask], p,
                             jac=lambda q: evaluate(q)[1][mask],
                             bounds=(lower, upper), x_scale=param_scale,
                             loss=loss, f_scale=scale, max_nfev=60,
                             gtol=1e-12, ftol=1e-9, xtol=1e-9).x

    all_rows = np.ones(len(y), dtype=bool)
    best_score = np.inf
    for p in starts:
        r = evaluate(p)[0]
        p = fit(p, all_rows, 'soft_l1', max(float(_mad(r)[0]), floor))
        r = evaluate(p)[0]
        p = fit(p, all_rows, 'soft_l1', max(1.5*float(_mad(r)[0]), floor))
        score = trimmed(evaluate(p)[0])
        if score < best_score:
            best, best_score = p, score

    # Once localized, discard stuck readings and refit the ordinary noise.
    for _ in range(3):
        r = evaluate(best)[0]
        threshold = max(4.5*float(_mad(r)[0]), 4*floor)
        inliers = np.abs(r - np.median(r)) <= threshold
        if np.count_nonzero(inliers) < max(5, int(0.8*len(y))):
            break
        updated = fit(best, inliers, 'linear', 1.)
        if np.allclose(updated, best, atol=1e-11, rtol=1e-8):
            best = updated
            break
        best = updated
    return {'source_xy': best[:2].tolist(),
            'amplitude': float(best[2] / green_scale),
            'junction_offset': float(best[3] + offset_ref)}
