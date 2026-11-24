"""Exponentially fitted transport and robust inversion of a Gaussian source."""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _face_rates(velocity, diffusivity, h):
    """Positive- and negative-direction coefficients of flux divided by h."""
    velocity = np.asarray(velocity, dtype=float)
    if diffusivity == 0:
        return np.maximum(velocity, 0) / h, np.maximum(-velocity, 0) / h
    z = np.abs(velocity) * (h / diffusivity)
    # B(z) = z exp(-z) / (1-exp(-z)), avoiding overflow and cancellation.
    ratio = np.ones_like(z)
    np.divide(z * np.exp(-z), -np.expm1(-z), out=ratio, where=z != 0)
    small = (diffusivity / h**2) * ratio
    return small + np.maximum(velocity, 0) / h, small + np.maximum(-velocity, 0) / h


def _operator(ux_face, uy_face, diffusivity, kappa, h):
    xp, xm = _face_rates(ux_face, diffusivity, h)
    yp, ym = _face_rates(uy_face, diffusivity, h)
    diagonal = xp[1:, :] + xm[:-1, :] + yp[:, 1:] + ym[:, :-1] + kappa
    ids = np.arange(diagonal.size).reshape(diagonal.shape)
    left, right = ids[:-1, :].ravel(), ids[1:, :].ravel()
    lower, upper = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    # Each boundary face contributes its center coefficient to the diagonal;
    # the ghost coefficient multiplies zero and has no matrix entry.
    return coo_matrix((
        np.concatenate((diagonal.ravel(), -xm[1:-1, :].ravel(),
                        -xp[1:-1, :].ravel(), -ym[:, 1:-1].ravel(),
                        -yp[:, 1:-1].ravel())),
        (np.concatenate((ids.ravel(), left, right, lower, upper)),
         np.concatenate((ids.ravel(), right, left, upper, lower))),
    ), shape=(ids.size, ids.size)).tocsc()


def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    """Solve the steady transport equation in [x-index, y-index] order."""
    source = np.asarray(source_map, dtype=float)
    matrix = _operator(ux_face, uy_face, diffusivity, kappa, h)
    return splu(matrix).solve(source.ravel()).reshape(source.shape)


class _TrimmedFit:
    """Profile nonnegative source strength, excluding the two worst sensors."""

    def __init__(self, observed):
        self.observed = observed
        self.rows = np.arange(observed.size)
        self.i, self.j = np.triu_indices(observed.size, 1)
        self.keep = (self.rows != self.i[:, None]) & (self.rows != self.j[:, None])
        y2 = observed**2
        self.y2 = y2.sum() - y2[self.i] - y2[self.j]

    def grid_loss(self, responses):
        # Evaluate all two-sensor omissions at once for each block of positions.
        z2 = responses**2
        zy = responses * self.observed
        norm = z2.sum(axis=1, keepdims=True) - z2[:, self.i] - z2[:, self.j]
        corr = zy.sum(axis=1, keepdims=True) - zy[:, self.i] - zy[:, self.j]
        np.maximum(corr, 0, out=corr)
        # Recompute nearly cancelled sums directly: the retained response can
        # be orders of magnitude weaker than the two omitted responses.
        weak = norm < 1e-10 * z2.sum(axis=1, keepdims=True)
        a, b = np.nonzero(weak)
        for start in range(0, a.size, 4096):
            s = slice(start, start + 4096)
            mask = self.keep[b[s]]
            norm[a[s], b[s]] = np.sum(z2[a[s]] * mask, axis=1)
            corr[a[s], b[s]] = np.maximum(np.sum(zy[a[s]] * mask, axis=1), 0)
        reduction = np.divide(corr**2, norm, out=np.zeros_like(norm), where=norm > 0)
        return np.min(self.y2 - reduction, axis=1)

    def best(self, response):
        # Direct sums avoid cancellation in both outlier removal and residuals.
        r = response * self.keep
        norm = np.einsum('ij,ij->i', r, r)
        corr = np.einsum('ij,j->i', r, self.observed)
        q = np.divide(np.maximum(corr, 0), norm, out=np.zeros_like(norm), where=norm > 0)
        residual = (q[:, None] * r - self.observed) * self.keep
        loss = np.einsum('ij,ij->i', residual, residual)
        k = np.argmin(loss)
        return loss[k], q[k], residual[k]


def invert_point_source(problem):
    """Estimate the center and peak amplitude of the Gaussian forcing."""
    n, h = problem['N'], problem['h']
    ij = np.asarray(problem['sensor_ij'], dtype=int)
    obs = np.asarray(problem['c_obs'], dtype=float)
    length = n * h
    width = problem['sigma_s'] / length
    # Normalize spatial parameters to [0,1] and concentration by a common
    # factor, without altering the least-squares optimum or sensor weights.
    cells = (np.arange(n) + 0.5) / n
    scale = np.max(np.abs(obs)) or 1.0
    fit = _TrimmedFit(obs / scale)
    lu = splu(_operator(problem['ux_face'], problem['uy_face'],
                        problem['D'], problem['kappa'], h))
    rhs = np.zeros((n*n, obs.size), order='F')
    rhs[ij[:, 0] * n + ij[:, 1], np.arange(obs.size)] = 1.0
    adjoint = np.ascontiguousarray(lu.solve(rhs, trans='T').T.reshape(-1, n, n))

    def evaluate(xy, strength=False):
        dx, dy = (cells[None, :] - np.asarray(xy)[:, None]) / width
        gx, gy = np.exp(-0.5 * dx**2), np.exp(-0.5 * dy**2)
        along_x = np.einsum('sij,i->sj', adjoint, gx)
        response = np.einsum('sj,j->s', along_x, gy)
        loss, q, residual = fit.best(response)
        if strength:
            return q * scale
        derivative_x = np.einsum('sij,i,j->s', adjoint, gx * dx, gy)
        derivative_y = np.einsum('sj,j->s', along_x, gy * dy)
        grad = (2 * q / width) * np.array((residual @ derivative_x, residual @ derivative_y))
        return loss, grad

    count = min(97, max(51, int(np.ceil(2.5 / width)) + 1))
    grid = np.linspace(0., 1., count)
    gauss = np.exp(-0.5 * ((cells[None, :] - grid[:, None]) / width)**2)
    intermediate = np.einsum('xi,sij->xsj', gauss, adjoint)
    responses = np.einsum('xsj,yj->xys', intermediate, gauss).reshape(-1, obs.size)
    score = np.concatenate([fit.grid_loss(responses[k:k+512])
                            for k in range(0, len(responses), 512)]).reshape(count, count)
    best_loss = np.inf
    starts = []
    for idx in np.argsort(score.ravel()):
        xy = grid[np.array(divmod(idx, count))]
        if any(np.max(np.abs(xy - start)) < max(width, 2.5 / (count-1)) for start in starts):
            continue
        starts.append(xy)
        result = minimize(evaluate, xy, jac=True, method='L-BFGS-B',
                          bounds=((0., 1.), (0., 1.)),
                          options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 80})
        if result.fun < best_loss:
            best_loss, best_xy = result.fun, result.x
        if len(starts) == 12:
            break
    return {'source_xy': (length * best_xy).tolist(),
            'strength_Q': float(evaluate(best_xy, strength=True))}
