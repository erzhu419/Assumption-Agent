import os
"""Conservative exponential face fluxes and robust Gaussian source inversion."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _bernoulli(z):
    """Stable evaluation including zero and the large positive tail."""
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    normal = (z != 0) & (z <= 50)
    out[normal] = z[normal] / np.expm1(z[normal])
    tail = z > 50
    e = np.exp(-z[tail])
    out[tail] = z[tail] * e / (1 - e)
    return out


def _operator(ux, uy, diffusivity, kappa, h):
    nx, ny = np.shape(uy)[0], np.shape(ux)[1]
    ids = np.arange(nx * ny).reshape(nx, ny)
    px = np.asarray(ux) * (h / diffusivity)
    py = np.asarray(uy) * (h / diffusivity)
    scale = diffusivity / h**2
    xm, xp = scale * _bernoulli(px), scale * _bernoulli(-px)
    ym, yp = scale * _bernoulli(py), scale * _bernoulli(-py)
    # The four outward cell contributions include the zero-ghost faces.
    diag = kappa + xp[1:, :] + xm[:-1, :] + yp[:, 1:] + ym[:, :-1]
    row = (ids, ids[:-1, :], ids[1:, :], ids[:, :-1], ids[:, 1:])
    col = (ids, ids[1:, :], ids[:-1, :], ids[:, 1:], ids[:, :-1])
    val = (diag, -xm[1:-1, :], -xp[1:-1, :], -ym[:, 1:-1], -yp[:, 1:-1])
    flat = lambda arrays: np.concatenate([a.ravel() for a in arrays])
    return coo_matrix((flat(val), (flat(row), flat(col))),
                      shape=(ids.size, ids.size)).tocsc()


def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    """Solve the cell-centered Scharfetter–Gummel balance."""
    source = np.asarray(source_map, dtype=float)
    lu = splu(_operator(ux_face, uy_face, diffusivity, kappa, h))
    return lu.solve(source.ravel()).reshape(source.shape)


def invert_point_source(problem):
    """Fit location and amplitude after profiling out the two faulty readings."""
    n, h = problem['N'], problem['h']
    L = n * h
    sigma = problem['sigma_s'] / L
    coord = (np.arange(n) + .5) / n
    observed = np.array(problem['c_obs'], dtype=float, copy=True).ravel()
    obs_scale = np.max(np.abs(observed))
    observed /= obs_scale if obs_scale else 1.0
    ij = np.asarray(problem['sensor_ij'], dtype=int)
    lu = splu(_operator(problem['ux_face'], problem['uy_face'], problem['D'],
                        problem['kappa'], h))
    m = observed.size
    rhs = np.zeros((n*n, m), order='F')
    rhs[ij[:, 0]*n + ij[:, 1], np.arange(m)] = 1
    g = np.ascontiguousarray(lu.solve(rhs, trans='T').T.reshape(m, n, n))

    a, b = np.triu_indices(m, k=1)
    retained = np.ones((a.size, m), dtype=bool)
    retained[np.arange(a.size), a] = False
    retained[np.arange(b.size), b] = False
    kept_power = np.sum(np.where(retained, observed**2, 0), axis=1)[:, None]

    prior_multiplier = float(os.environ.get('PECLET_PRIOR_TEST', '2'))
    prior_scale = 1.
    prior_weight = 0.

    def grid_objective(response):
        # Evaluate the exact nonnegative LS amplitude on each retained subset.
        sq = response**2
        cross = response * observed[:, None]
        norm = sq.sum(axis=0) - sq[a] - sq[b]
        inner = np.maximum(cross.sum(axis=0) - cross[a] - cross[b], 0)
        reduction = np.divide(inner**2, norm, out=np.zeros_like(norm), where=norm > 0)
        rss = kept_power - reduction
        if prior_weight:
            amp = np.divide(inner, norm, out=np.full_like(norm, np.inf), where=norm > 0.)
            rss += prior_weight * np.log1p(amp / prior_scale)
        return np.min(rss, axis=0)

    resolution = max(61, min(141, int(np.ceil(4 / sigma)) + 1))
    grid = np.linspace(0., 1., resolution)
    kernel = np.exp(-0.5*((grid[:, None] - coord) / sigma)**2)
    left = np.einsum('ai,mij->amj', kernel, g)
    response = np.einsum('amj,bj->mab', left, kernel).reshape(m, -1)
    def screen():
        return np.concatenate([grid_objective(response[:, j:j+512])
                               for j in range(0, response.shape[1], 512)])
    score = screen()
    lower_strength = max(np.sort(observed)[-3], 0.) / response.max()
    prior_scale = max(lower_strength, np.finfo(float).tiny)
    prior_weight = prior_multiplier * max(score.min(), 0.) / (m - 5)
    score = screen()

    def evaluate(z):
        d = (coord - z[:, None]) / sigma
        e = np.exp(-0.5*d**2)
        ex, ey = e
        dx, dy = e*d/sigma
        gy = np.einsum('mij,j->mi', g, ey)
        val = gy @ ex
        v = np.where(retained, val, 0.)
        norm = np.einsum('ij,ij->i', v, v)
        inner = np.maximum(v @ observed, 0.)
        amp = np.divide(inner, norm, out=np.zeros_like(norm), where=norm > 0)
        residual = np.where(retained, amp[:, None]*val - observed, 0.)
        losses = np.einsum('ij,ij->i', residual, residual)
        if prior_weight:
            losses += prior_weight * np.log1p(amp / prior_scale)
        k = np.argmin(losses)
        jac = np.array([gy @ dx, np.einsum('mij,i,j->m', g, ex, dy)])
        grad = 2 * amp[k] * (jac @ residual[k])
        if prior_weight and amp[k] > 0.:
            damp = (jac @ np.where(retained[k], observed - 2 * amp[k] * val, 0.)) / norm[k]
            grad += prior_weight / (prior_scale + amp[k]) * damp
        return losses[k], grad, amp[k]

    starts = []
    for i in np.argsort(score):
        z = grid[np.array(divmod(i, resolution))]
        if all(np.max(np.abs(z - u)) >= max(sigma*.6, 2/(resolution-1)) for u in starts):
            starts.append(z)
            if len(starts) == 10:
                break
    best = (np.inf, None, 0.)
    for z in starts:
        fit = minimize(lambda x: evaluate(x)[:2], z, method='L-BFGS-B',
                       jac=True, bounds=((0., 1.), (0., 1.)),
                       options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 65})
        loss, _, amp = evaluate(fit.x)
        if loss < best[0]:
            best = loss, fit.x, amp
    return {'source_xy': (L*best[1]).tolist(), 'strength_Q': float(obs_scale*best[2])}
