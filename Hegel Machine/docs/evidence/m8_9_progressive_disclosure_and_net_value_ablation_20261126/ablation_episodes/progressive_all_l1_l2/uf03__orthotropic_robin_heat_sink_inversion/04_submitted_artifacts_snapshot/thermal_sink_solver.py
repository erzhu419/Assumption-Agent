"""Conservative finite-volume source inversion from point temperatures."""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu


def _sensor_green(k, gamma, sensor):
    n = k.shape[0]
    ids = np.arange(n*n).reshape(n, n)
    u = np.r_[ids[:-1, :].ravel(), ids[:, :-1].ravel()]
    v = np.r_[ids[1:, :].ravel(), ids[:, 1:].ravel()]
    a, b = k.ravel()[u], k.ravel()[v]
    w = 2*a*b/(a+b)
    edge = np.zeros_like(k)
    edge[[0, -1], :] += gamma/n
    edge[:, [0, -1]] += gamma/n
    diag = edge.ravel() + np.bincount(u, w, n*n) + np.bincount(v, w, n*n)
    idx = ids.ravel()
    matrix = sparse.csc_matrix((np.r_[diag, -w, -w], (np.r_[idx, u, v], np.r_[idx, v, u])), shape=(n*n, n*n))
    factor = splu(matrix)
    sensors, inverse = np.unique(ids[tuple(sensor.T)], return_inverse=True)
    green = np.empty((sensors.size, n*n))
    for start in range(0, sensors.size, 16):
        loc = sensors[start:start+16]
        rhs = np.zeros((n*n, loc.size), order='F')
        rhs[loc, np.arange(loc.size)] = 1.0/n**2
        green[start:start+loc.size] = factor.solve(rhs, trans='T').T
    return green[inverse]


def _affine(pred, obs, mask):
    count = mask.sum(axis=1)
    x = (pred*mask).sum(axis=1)/count
    y = (obs*mask).sum(axis=1)/count
    dx = pred - x[:, None]
    dy = obs - y[:, None]
    var = (mask*dx**2).sum(axis=1)
    amp = np.maximum((mask*dx*dy).sum(axis=1)/np.maximum(var, np.finfo(float).tiny), 0)
    return amp, y-amp*x


def localize_thermal_source(measurement: dict) -> dict:
    """Estimate source coordinates, positive amplitude, and common sensor offset."""
    n = int(measurement['N'])
    sigma = float(measurement['sigma_m'])
    sensor = np.asarray(measurement['sensor_ij'], dtype=int)
    obs = np.asarray(measurement['T_obs'], dtype=float)
    green = _sensor_green(np.asarray(measurement['kappa'], dtype=float), float(measurement['gamma_robin']), sensor)
    axis = (np.arange(n)+0.5)/n
    x, y = (a.ravel() for a in np.meshgrid(axis, axis, indexing='ij'))
    grid = np.linspace(0, 1, min(41, max(17, int(np.ceil(1.25/sigma))+1)))
    gx, gy = (a.ravel() for a in np.meshgrid(grid, grid, indexing='ij'))
    points = np.column_stack((gx, gy))
    inv = 1/sigma**2
    m = obs.size
    keep = max(4, m-max(1, int(np.ceil(0.10*m))))
    score = np.empty(len(points))
    amps = np.empty(len(points))
    offsets = np.empty(len(points))
    for start in range(0, len(points), 32):
        end = min(start+32, len(points))
        q = np.exp(-0.5*inv*((gx[start:end,None]-x)**2 + (gy[start:end,None]-y)**2))
        pred = np.einsum('ij,kj->ik', q, green)
        mask = np.ones(pred.shape, dtype=bool)
        for _ in range(5):
            amp, off = _affine(pred, obs, mask)
            sq = (amp[:,None]*pred + off[:,None] - obs)**2
            ii = np.argpartition(sq, keep-1, axis=1)[:, :keep]
            mask.fill(False)
            np.put_along_axis(mask, ii, True, axis=1)
        score[start:end] = np.sum(np.where(mask, sq, 0), axis=1)
        amps[start:end], offsets[start:end] = amp, off

    def evaluate(p, ids=None, jac=False):
        g = green if ids is None else green[ids]
        dx, dy = x-p[0], y-p[1]
        q = np.exp(-0.5*inv*(dx*dx+dy*dy))
        f = np.einsum('ij,j->i', g, q)
        if jac:
            return np.column_stack((np.einsum('ij,j->i', g, (p[2]*inv)*dx*q),
                                    np.einsum('ij,j->i', g, (p[2]*inv)*dy*q),
                                    f, np.ones_like(f)))
        return p[2]*f+p[3]-(obs if ids is None else obs[ids])

    bounds = ([0, 0, 0, -np.inf], [1, 1, np.inf, np.inf])

    def fit(p, ids=None, **options):
        return least_squares(lambda z: evaluate(z, ids), p, jac=lambda z: evaluate(z, ids, True),
                             bounds=bounds, x_scale='jac', max_nfev=50, **options).x

    best = None
    best_score = np.inf
    starts = []
    for idx in np.argsort(score):
        xy = points[idx]
        if any(np.linalg.norm(xy-prev) < 1.75/(grid.size-1) for prev in starts):
            continue
        starts.append(xy)
        p = np.r_[xy, max(amps[idx], 1e-8), offsets[idx]]
        prev = None
        for _ in range(8):
            sq = evaluate(p)**2
            ids = np.sort(np.argpartition(sq, keep-1)[:keep])
            if prev is not None and np.array_equal(prev, ids):
                break
            p = fit(p, ids)
            prev = ids
        obj = np.partition(evaluate(p)**2, keep-1)[:keep].sum()
        if obj < best_score:
            best, best_score = p, obj
        if len(starts) == 7:
            break
    r = evaluate(best)
    noise = 1.4826*np.median(np.abs(r-np.median(r)))
    scale = max(2.5*noise, 50*np.finfo(float).eps*np.max(np.abs(obs)), np.finfo(float).eps)
    p = fit(best, loss='cauchy', f_scale=scale, gtol=1e-11)
    # Refit the clean readings without shrinking ordinary noise along weakly
    # identified directions, such as amplitude versus distance to an edge.
    r = evaluate(p)
    deviation = np.abs(r - np.median(r))
    cutoff = max(5 * 1.4826 * np.median(deviation), scale)
    ids = np.flatnonzero(deviation <= cutoff)
    if keep <= ids.size < m:
        p = fit(p, ids, gtol=1e-11)
    return {'source_xy': [float(p[0]), float(p[1])], 'amplitude': float(p[2]), 'junction_offset': float(p[3])}
