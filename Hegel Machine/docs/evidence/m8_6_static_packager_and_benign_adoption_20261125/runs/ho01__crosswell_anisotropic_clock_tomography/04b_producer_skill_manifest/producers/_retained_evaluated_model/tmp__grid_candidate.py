"""Invert straight-ray crosswell picks for a Gaussian plume and receiver clocks."""

import os
from profile_grid import location
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import brentq, least_squares, minimize
from scipy.special import erf


class _Rays:
    def __init__(self, width, radius, sources, receivers):
        dz = receivers[None, :] - sources[:, None]
        self.length = np.hypot(width, dz).ravel()
        self.c = width / self.length
        self.s = dz.ravel() / self.length
        self.z = np.broadcast_to(sources[:, None], dz.shape).ravel()
        self.r = radius
        self.bg = np.column_stack((self.length, -self.length * self.s**2))

    def plume(self, x, z, derivatives=False):
        u = self.c * x + self.s * (z - self.z)
        v = -self.s * x + self.c * (z - self.z)
        end = self.length - u
        r = self.r
        cross = np.exp(-0.5 * (v / r)**2)
        span = (erf(end / (np.sqrt(2) * r)) + erf(u / (np.sqrt(2) * r)))
        k = r * np.sqrt(np.pi / 2) * cross * span
        if not derivatives:
            return k
        du = cross * (np.exp(-0.5 * (u / r)**2) - np.exp(-0.5 * (end / r)**2))
        dv = -v * k / r**2
        return k, self.c * du - self.s * dv, self.s * du + self.c * dv


def _linear(a, y, weights):
    return np.linalg.lstsq(a * weights[:, None], y * weights, rcond=None)[0]


def _seeds(rays, a, y, width, height):
    # Adapt Cauchy reweighting to scalar picks. Remove clock and background
    # components from every plume hypothesis before scoring its ray pattern.
    scale = 0.35  # milliseconds; broad enough to retain signal under the null
    weights = np.ones_like(y)
    for _ in range(9):
        beta = _linear(a, y, weights)
        weights = 1.0 / np.hypot(1, (y - a @ beta) / scale)
    q = np.linalg.qr(a * weights[:, None], mode='reduced')[0]
    wy = y * weights
    residual = wy - q @ (q.T @ wy)
    step = 0.55 * rays.r
    xs = np.linspace(0, width, int(np.ceil(width / step)) + 1)
    zs = np.linspace(0, height, int(np.ceil(height / step)) + 1)
    x, z = (t.ravel() for t in np.meshgrid(xs, zs))
    score = np.empty_like(x)
    for i in range(0, len(x), 256):
        sl = slice(i, i + 256)
        k = rays.plume(x[sl, None], z[sl, None]) * weights
        projection = np.einsum('ij,jk->ik', k, q)
        norm = np.sqrt(np.maximum(1e-20, np.sum(k*k, axis=1) - np.sum(projection**2, axis=1)))
        score[sl] = np.maximum(0, np.einsum('ij,j->i', k, residual)) / norm
    grid = score.reshape(len(zs), len(xs))
    peaks = np.flatnonzero((grid == maximum_filter(grid, size=3)).ravel() & (score > 0))
    if not len(peaks):
        peaks = np.array([np.argmax(score)])
    best = peaks[np.argsort(-score[peaks])[:8]]
    return zip(x[best], z[best]), weights


def _bounded_clocks(means, counts):
    # Profile the clock nuisance exactly: the shared multiplier enforces the
    # zero-sum gauge even if one or more clocks are on the box boundary.
    def clocks(multiplier):
        return np.clip(means - multiplier / counts, -4., 4.)

    offset = np.sum(means) / np.sum(1 / counts)
    c = means - offset / counts
    if np.max(np.abs(c)) <= 4.:
        return c
    root = brentq(lambda t: np.sum(clocks(t)),
                  np.min((means - 4.) * counts), np.max((means + 4.) * counts))
    return clocks(root)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in metres and receiver clock biases in seconds."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    y = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000.
    rays = _Rays(width, float(survey.get('R_plume_m', 5.5)), src, rcv)
    clock = np.eye(len(rcv))[:, :-1]
    clock[-1] = -1
    a = np.column_stack((rays.bg, np.tile(clock, (len(src), 1))))
    seeds, weights = _seeds(rays, a, y, width, height)

    # p = (plume x, plume z, amplitude, background coefficients, free clocks).
    def residual(p):
        return a @ p[3:] + p[2] * rays.plume(*p[:2]) - y

    def jacobian(p):
        k, kx, kz = rays.plume(*p[:2], derivatives=True)
        return np.column_stack((p[2] * kx, p[2] * kz, k, a))

    best = None
    for x, z in seeds:
        beta = _linear(np.column_stack((rays.plume(x, z), a)), y, weights)
        beta[0] = max(beta[0], 0.001)
        fit = least_squares(residual, np.r_[x, z, beta], jac=jacobian,
                            bounds=(np.r_[0., 0., 0., np.full(a.shape[1], -np.inf)],
                                    np.r_[width, height, np.full(a.shape[1]+1, np.inf)]),
                            x_scale='jac', loss='cauchy', f_scale=0.18, max_nfev=90)
        if best is None or fit.cost < best.cost:
            best = fit

    y2 = y.reshape(len(src), len(rcv))
    p = best.x[:5].copy()
    p[3] = max(p[3], 0.)
    keep = (abs(best.fun) < 2.5).reshape(y2.shape)
    # A clock needs at least one usable pick. Retain the closest pick in a
    # receiver column if it has no residual inside the inlier gate.
    for j in np.flatnonzero(~keep.any(axis=0)):
        keep[np.argmin(abs(best.fun.reshape(y2.shape)[:, j])), j] = True
    counts = keep.sum(axis=0)

    def profile(p, clocks_only=False):
        k, kx, kz = rays.plume(*p[:2], derivatives=True)
        e = y2 - (p[2] * k + rays.bg @ p[3:]).reshape(y2.shape)
        c = _bounded_clocks(np.sum(np.where(keep, e, 0.), axis=0) / counts, counts)
        if clocks_only:
            return c
        e = np.where(keep, e - c, 0.).ravel()
        grad = -np.r_[p[2]*(e @ kx), p[2]*(e @ kz), e @ k, e @ rays.bg]
        return 0.5 * (e @ e), grad

    scales = np.array([rays.r, rays.r, 0.08, 0.4, 0.08])

    def objective(q):
        loss, gradient = profile(q * scales)
        return loss, gradient * scales

    final = minimize(objective, p / scales, jac=True, method='L-BFGS-B',
                     bounds=((0, width/scales[0]), (0, height/scales[1]),
                             (0, None), (0, None), (None, None)),
                     options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 160})
    p = final.x * scales
    xz = location(rays,a,y,keep,p,width,height,float(os.environ['PRIOR_WIDTH']), os.environ['MARGINAL']=='1')
    p[:2] = xz
    key = (rays.plume(*xz), rays.bg[:,0], rays.bg[:,1])
    linear_fit = minimize(lambda q: (objective(np.r_[xz/scales[:2],q])[0], objective(np.r_[xz/scales[:2],q])[1][2:]), p[2:]/scales[2:], jac=True, method='L-BFGS-B', bounds=((0,None),(0,None),(None,None)), options={'ftol':1e-13, 'gtol':1e-9})
    p[2:] = linear_fit.x*scales[2:]
    return {'plume_xz_m': p[:2].tolist(),
            'rcv_clock_bias_s': (profile(p, clocks_only=True) / 1000.).tolist()}
