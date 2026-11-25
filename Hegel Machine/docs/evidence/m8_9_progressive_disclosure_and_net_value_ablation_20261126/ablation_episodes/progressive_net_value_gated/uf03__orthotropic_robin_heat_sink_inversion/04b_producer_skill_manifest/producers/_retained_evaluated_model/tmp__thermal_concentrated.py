"""Recover a Gaussian heat source and a common thermistor offset."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _factor_operator(kappa, gamma):
    n = kappa.shape[0]
    cells = np.arange(n*n).reshape(n, n)
    left = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    right = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    k = kappa.ravel()
    conductance = 2*k[left]*k[right]/(k[left]+k[right])
    diagonal = np.zeros((n, n))
    diagonal[[0, -1], :] += gamma/n
    diagonal[:, [0, -1]] += gamma/n
    diagonal = diagonal.ravel()
    if n == 1:
        diagonal[0] = 4*gamma
    diagonal += np.bincount(left, conductance, minlength=n*n)
    diagonal += np.bincount(right, conductance, minlength=n*n)
    ids = cells.ravel()
    matrix = coo_matrix((np.r_[diagonal, -conductance, -conductance],
                         (np.r_[ids, left, right], np.r_[ids, right, left])),
                        shape=(n*n, n*n)).tocsc()
    return splu(matrix)


def _trimmed_regression(pred, target):
    """Rank all grid points by trimmed residual energy after profiling gain/offset."""
    m = len(target)
    keep = max(min(4, m), m-int(np.ceil(0.12*m)))
    weights = np.ones_like(pred, dtype=bool)
    for _ in range(5):
        count = weights.sum(axis=1)
        mean_p = (weights*pred).sum(axis=1)/count
        mean_y = (weights*target).sum(axis=1)/count
        pc = pred-mean_p[:, None]
        numerator = (weights*pc*(target-mean_y[:, None])).sum(axis=1)
        denominator = (weights*pc**2).sum(axis=1)
        a = np.maximum(numerator/np.maximum(denominator, np.finfo(float).tiny), 0)
        b = mean_y-a*mean_p
        residual = pred*a[:, None]+b[:, None]-target
        indices = np.argpartition(np.abs(residual), keep-1, axis=1)[:, :keep]
        weights.fill(False)
        np.put_along_axis(weights, indices, True, axis=1)
    return (weights*residual**2).sum(axis=1), a, b


def localize_thermal_source(measurement: dict) -> dict:
    n = int(measurement['N'])
    kappa = np.asarray(measurement['kappa'], dtype=float)
    gamma = float(measurement['gamma_robin'])
    sigma = float(measurement['sigma_m'])
    ij = np.asarray(measurement['sensor_ij'], dtype=int)
    y = np.asarray(measurement['T_obs'], dtype=float)
    cells, inverse = np.unique(n*ij[:, 0]+ij[:, 1], return_inverse=True)
    rhs = np.zeros((n*n, len(cells)))
    rhs[cells, np.arange(len(cells))] = 1/n**2
    green = np.ascontiguousarray(_factor_operator(kappa, gamma).solve(rhs).T)

    centers = (np.arange(n)+0.5)/n
    axis = np.linspace(0, 1, 25)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    gr = np.exp(-0.5*((axis[:, None]-centers)/sigma)**2)
    source = (gr[:, None, :, None]*gr[None, :, None, :]).reshape(len(grid), n*n)
    pred = (source @ green.T)[:, inverse]
    scores, amplitudes, offsets = _trimmed_regression(pred, y)
    seeds = []
    for idx in np.argsort(scores):
        if all(np.linalg.norm(grid[idx]-grid[s]) >= 0.115 for s in seeds):
            seeds.append(idx)
            if len(seeds) == 6:
                break

    x = np.repeat(centers, n)
    z = np.tile(centers, n)
    inv_s2 = 1/sigma**2
    amplitude_scale = max(amplitudes[seeds[0]], np.std(y)/max(np.max(np.ptp(pred, axis=1)), 1e-20), 1e-12)
    lo = np.array([0., 0., 0., -np.inf])
    hi = np.array([1., 1., np.inf, np.inf])

    def evaluate(p):
        xs, ys, scaled_a, b = p
        a = scaled_a*amplitude_scale
        dx, dy = x-xs, z-ys
        g = np.exp(-0.5*inv_s2*(dx*dx+dy*dy))
        basis = np.array([g, g*dx, g*dy])
        f, fx, fy = np.einsum('ij,kj->ki', green, basis)[..., inverse]
        residual = a*f+b-y
        jac = np.column_stack((a*inv_s2*fx, a*inv_s2*fy, amplitude_scale*f, np.ones_like(f)))
        return residual, jac

    # Use a common initial cutoff to compare basins, then estimate its noise
    # component again after continuous localization removes grid mismatch.
    best = seeds[0]
    r0 = amplitudes[best]*pred[best]+offsets[best]-y
    floor = max(1.e-12, 1.e-6*np.ptp(y))

    def cutoff(r):
        return max(floor, 2.5*1.4826*np.median(np.abs(r-np.median(r))))

    def fit(p, scale):
        return least_squares(lambda q: evaluate(q)[0], p,
                             jac=lambda q: evaluate(q)[1], bounds=(lo, hi),
                             loss='cauchy', f_scale=scale, x_scale='jac',
                             ftol=1.e-9, max_nfev=90)

    common = cutoff(r0)
    fits = [fit(np.r_[grid[s], max(amplitudes[s]/amplitude_scale, 1.e-5), offsets[s]], common)
            for s in seeds]
    winner = min(fits, key=lambda f: f.cost)
    for _ in range(2):
        winner = fit(winner.x, cutoff(winner.fun))
    # Concentrate the fit on the expected clean majority before estimating
    # the noise scale. A stuck sensor can otherwise partly mask its own
    # residual by pulling a weakly constrained fit toward itself.
    keep = max(5, len(y)-max(1, round(0.06*len(y))))

    def subset_fit(p, mask, loss='linear', scale=1.):
        return least_squares(lambda q: evaluate(q)[0][mask], p,
                             jac=lambda q: evaluate(q)[1][mask],
                             bounds=(lo, hi), x_scale='jac', loss=loss,
                             f_scale=scale, ftol=1.e-10, max_nfev=90)

    if keep < len(y):
        def concentrate(start):
            p, r = start.x, start.fun
            last = None
            for _ in range(6):
                ids = np.sort(np.argpartition(np.abs(r), keep-1)[:keep])
                if last is not None and np.array_equal(ids,last):
                    break
                part = subset_fit(p, ids)
                p, r = part.x, evaluate(part.x)[0]
                last = ids
            return np.partition(r*r, keep-1)[:keep].sum(), p, r

        _, p, r = min((concentrate(f) for f in [winner, *sorted(fits, key=lambda f:f.cost)[:3]]),
                      key=lambda item: item[0])
        inlier = np.abs(r-np.median(r)) <= 2.0*cutoff(r)
        if np.count_nonzero(inlier) <= 4:
            inlier = np.ones(len(y), dtype=bool)
        winner = subset_fit(p, inlier, 'huber', cutoff(r))
    else:
        inlier = np.ones(len(y), dtype=bool)
    xs, ys, a, b = winner.x

    # On a weakly constrained wall-normal ridge, summarize likelihood over
    # physical source positions instead of selecting the lowest-residual point.
    residual, jac = evaluate(winner.x)
    j = jac[inlier]
    sd = np.linalg.norm(residual[inlier]) / np.sqrt(max(len(j)-4, 1))
    cov = sd**2 * np.linalg.pinv(j.T @ j, rcond=1e-13)
    pos_sd = np.sqrt(np.maximum(np.diag(cov)[:2], 0.0))
    xy = np.array([xs, ys])
    if sd > floor and np.any(np.minimum(xy, 1-xy) < 2.5*pos_sd):
        radius = np.clip(6*pos_sd, 0.008, 0.40)
        low, high = np.maximum(0., xy-radius), np.minimum(1., xy+radius)
        gx, gy = (np.linspace(l, h, 81) for l,h in zip(low,high))
        xx = np.exp(-0.5*((gx[:,None]-centers)/sigma)**2)
        yy = np.exp(-0.5*((gy[:,None]-centers)/sigma)**2)
        # Evaluate the separable Gaussian without a dense (81**2)-by-N**2 array.
        gi = green[inverse[inlier]].reshape(-1,n,n)
        f = ((xx @ gi) @ yy.T).reshape(len(gi),-1).T
        data = y[inlier]
        fm = f.mean(axis=1)
        fc = f-fm[:,None]
        norm2 = np.einsum('ij,ij->i',fc,fc)
        amp = (fc @ (data-data.mean()))/np.maximum(norm2, np.finfo(float).tiny)
        off = data.mean()-amp*fm
        rss = np.sum((f*amp[:,None]+off[:,None]-data)**2,axis=1)
        logw = -0.5 * rss/(sd*sd)
        logw[(amp <= 0.) | (norm2 <= 0.)] = -np.inf
        w = np.exp(logw-np.max(logw)).reshape(len(gx),len(gy))
        w[[0,-1],:] *= 0.5
        w[:,[0,-1]] *= 0.5
        w /= w.sum()
        xs, ys = w.sum(axis=1) @ gx, w.sum(axis=0) @ gy
        a = np.exp(w.ravel() @ np.log(np.maximum(amp,np.finfo(float).tiny)))/amplitude_scale
        b = w.ravel() @ off
    return {'source_xy': [float(xs), float(ys)],
            'amplitude': float(a*amplitude_scale), 'junction_offset': float(b)}
