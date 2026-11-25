"""Localize a Gaussian heat source from sparse plate thermistors."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _plate_operator(kappa, gamma):
    n = kappa.shape[0]
    cells = np.arange(n*n).reshape(n, n)
    left = np.r_[cells[:-1, :].ravel(), cells[:, :-1].ravel()]
    right = np.r_[cells[1:, :].ravel(), cells[:, 1:].ravel()]
    k = kappa.ravel()
    g = 2*k[left]*k[right]/(k[left]+k[right])
    boundary = np.r_[cells[0], cells[-1], cells[:, 0], cells[:, -1]]
    diag = (np.bincount(left, g, minlength=n*n) +
            np.bincount(right, g, minlength=n*n) +
            gamma/n*np.bincount(boundary, minlength=n*n))
    return coo_matrix((np.r_[-g, -g, diag],
                       (np.r_[left, right, cells.ravel()],
                        np.r_[right, left, cells.ravel()])),
                      shape=(n*n, n*n)).tocsc()


def _response_bank(measurement):
    n = int(measurement['N'])
    sensor = np.asarray(measurement['sensor_ij'], dtype=int)
    cells, inverse = np.unique(sensor[:, 0]*n+sensor[:, 1], return_inverse=True)
    lu = splu(_plate_operator(np.asarray(measurement['kappa'], dtype=float),
                              float(measurement['gamma_robin'])))
    bank = np.empty((len(cells), n*n))
    rhs = np.zeros(n*n)
    for row, cell in enumerate(cells):
        rhs[cell] = 1.0/(n*n)
        bank[row] = lu.solve(rhs, trans='T')
        rhs[cell] = 0.0
    return bank[inverse].reshape(-1, n, n)


def _coarse_fit(templates, y, keep):
    w = np.ones_like(templates)
    for step in range(12):
        count = w.sum(axis=1)
        xbar = np.sum(w*templates, axis=1)/count
        ybar = np.sum(w*y, axis=1)/count
        x = templates-xbar[:, None]
        a = np.maximum(0.0, np.sum(w*x*(y-ybar[:, None]), axis=1) /
                       np.maximum(np.sum(w*x*x, axis=1), np.finfo(float).tiny))
        b = ybar-a*xbar
        r = a[:, None]*templates+b[:, None]-y
        if step < 6:
            center = np.median(r, axis=1, keepdims=True)
            mad = np.median(np.abs(r-center), axis=1, keepdims=True)
            w = np.minimum(1.0, np.maximum(1.6*mad, 1e-14)/
                           np.maximum(np.abs(r), 1e-14))
        else:
            ix = np.argpartition(np.abs(r), keep-1, axis=1)[:, :keep]
            w.fill(0.0)
            np.put_along_axis(w, ix, 1.0, axis=1)
    score = np.sort(r*r, axis=1)[:, :keep].sum(axis=1)
    return score, a, b


def localize_thermal_source(measurement: dict) -> dict:
    """Return source coordinates, volumetric amplitude and shared offset."""
    y = np.asarray(measurement['T_obs'], dtype=float)
    sigma = float(measurement['sigma_m'])
    n = int(measurement['N'])
    bank = _response_bank(measurement)
    coord = (np.arange(n)+0.5)/n
    grid = np.linspace(0.0, 1.0, int(np.clip(np.ceil(1.8/sigma)+1, 19, 49)))
    e = np.exp(-0.5*((coord[None, :]-grid[:, None])/sigma)**2)
    mids = np.einsum('mij,xi->mxj', bank, e)
    templates = np.einsum('mxj,yj->xym', mids, e).reshape(-1, y.size)
    xy = np.stack(np.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(-1, 2)
    keep = max(4, y.size-int(np.ceil(0.10*y.size)))
    scores, amplitude, offset = _coarse_fit(templates, y, keep)

    def residual_and_jac(p):
        x, z, a, b = p
        dx, dz = coord-x, coord-z
        ex, ez = np.exp(-0.5*(dx/sigma)**2), np.exp(-0.5*(dz/sigma)**2)
        u = np.einsum('mij,j->mi', bank, ez)
        uz = np.einsum('mij,j->mi', bank, ez*dz/sigma**2)
        t = np.einsum('mi,i->m', u, ex)
        tx = np.einsum('mi,i->m', u, ex*dx/sigma**2)
        tz = np.einsum('mi,i->m', uz, ex)
        return a*t+b-y, np.column_stack((a*tx, a*tz, t, np.ones_like(y)))

    bounds = ([0.0, 0.0, 0.0, -np.inf], [1.0, 1.0, np.inf, np.inf])

    def trimmed(p, jac=False):
        r, j = residual_and_jac(p)
        ix = np.argpartition(np.abs(r), keep-1)[keep:]
        if jac:
            j[ix] = 0.0
            return j
        r[ix] = 0.0
        return r

    starts = []
    best = None
    for i in np.argsort(scores):
        if any(np.linalg.norm(xy[i]-s) < max(2.1*(grid[1]-grid[0]), sigma) for s in starts):
            continue
        starts.append(xy[i])
        p0 = np.r_[xy[i], max(amplitude[i], 1e-8), offset[i]]
        fit = least_squares(trimmed, p0, jac=lambda p: trimmed(p, True),
                            bounds=bounds, x_scale='jac', max_nfev=60,
                            ftol=1e-10, gtol=1e-12)
        if best is None or fit.cost < best.cost:
            best = fit
        if len(starts) == 8:
            break

    # Trimming can discard the clean readings nearest a boundary source.
    # Reintroduce all readings with bounded residual influence before deciding
    # which sensors to exclude from the final least-squares polish.
    def noise_scale(r):
        return max(1.4826*np.median(np.abs(r-np.median(r))), 1e-12)

    scale = noise_scale(residual_and_jac(best.x)[0])
    robust = least_squares(lambda q: residual_and_jac(q)[0], best.x,
                           jac=lambda q: residual_and_jac(q)[1],
                           bounds=bounds, x_scale='jac', loss='huber',
                           f_scale=1.5*scale, max_nfev=100, gtol=1e-12)
    p = robust.x
    r, _ = residual_and_jac(p)
    scale = noise_scale(r)
    for _ in range(3):
        good = np.abs(r-np.median(r)) <= 4.25*scale
        if np.count_nonzero(good) < keep:
            good = np.argpartition(np.abs(r), keep-1)[:keep]
        fit = least_squares(lambda q: residual_and_jac(q)[0][good], p,
                            jac=lambda q: residual_and_jac(q)[1][good],
                            bounds=bounds, x_scale='jac', gtol=1e-12)
        p = fit.x
        r, _ = residual_and_jac(p)
    return {'source_xy': [float(p[0]), float(p[1])],
            'amplitude': float(p[2]), 'junction_offset': float(p[3])}
