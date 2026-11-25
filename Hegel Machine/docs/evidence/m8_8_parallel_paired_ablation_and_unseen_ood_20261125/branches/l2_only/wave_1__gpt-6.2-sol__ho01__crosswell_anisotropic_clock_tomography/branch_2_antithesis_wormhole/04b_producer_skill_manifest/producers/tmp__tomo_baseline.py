"""Straight-ray inversion of cross-well first arrivals."""
import numpy as np
from scipy.linalg import helmert, qr
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


class _Rays:
    def __init__(self, width, source, receiver, radius):
        dz = receiver[None, :] - source[:, None]
        length = np.hypot(width, dz)
        self.ux = (width / length).ravel()
        self.uz = (dz / length).ravel()
        self.source = np.broadcast_to(source[:, None], dz.shape).ravel()
        self.length = length.ravel()
        self.radius = radius
        self.clocks = helmert(receiver.size, full=False).T
        self.design = np.column_stack((self.length, -self.length * self.uz**2,
                                       np.tile(self.clocks, (source.size, 1))))

    def plume(self, x, z, derivative=False):
        r = self.radius
        dz = z - self.source
        t = (self.ux * x + self.uz * dz) / r
        p = (self.ux * dz - self.uz * x) / r
        end = self.length / r - t
        envelope = np.exp(-0.5 * p**2)
        path = np.sqrt(2 * np.pi) * r * envelope * (ndtr(end) - ndtr(-t))
        if not derivative:
            return path
        along = envelope * (np.exp(-0.5 * t**2) - np.exp(-0.5 * end**2))
        normal = -p / r * path
        return path, along * self.ux - normal * self.uz, along * self.uz + normal * self.ux


def _linear(design, y, weights):
    return np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]


def _fit(survey):
    w, h = float(survey['W_m']), float(survey['H_m'])
    src, rcv = (np.asarray(survey[k], dtype=float) for k in ('z_src_m', 'z_rcv_m'))
    rays = _Rays(w, src, rcv, float(survey.get('R_plume_m', 5.5)))
    y = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    a = rays.design
    lo = np.r_[0., 0., 0., np.full(a.shape[1], -np.inf)]
    hi = np.r_[w, h, np.full(a.shape[1] + 1, np.inf)]

    def prediction(p):
        return a @ p[3:] + p[2] * rays.plume(*p[:2])

    def jac(p):
        g, gx, gz = rays.plume(*p[:2], derivative=True)
        return np.column_stack((p[2] * gx, p[2] * gz, g, a))

    def refine(p, target, keep, scale=None):
        return least_squares(lambda b: (prediction(b) - target)[keep], p,
                             jac=lambda b: jac(b)[keep], bounds=(lo, hi),
                             x_scale='jac', loss='linear' if scale is None else 'soft_l1',
                             f_scale=1. if scale is None else scale,
                             max_nfev=60, ftol=1e-9).x

    bg = _linear(a, y, np.ones(y.size))
    for _ in range(12):
        e = y - a @ bg
        weights = (1 + (e / .38)**2)**(-.25)
        bg = _linear(a, y, weights)

    # Project both data and candidate plume signatures through the same
    # weighted background/clock null space before scoring the coarse grid.
    q = qr(a * weights[:, None], mode='economic')[0]
    gy = y * weights
    gy -= q @ (q.T @ gy)
    d = rays.radius / 2.5
    xy = np.array([(x, z) for x in np.linspace(0, w, int(np.ceil(w/d))+1)
                   for z in np.linspace(0, h, int(np.ceil(h/d))+1)])
    paths = rays.plume(xy[:, :1], xy[:, 1:]).T
    g = paths * weights[:, None]
    g -= q @ (q.T @ g)
    dots = gy @ g
    norm = np.maximum(np.sum(g*g, axis=0), 1e-30)
    merit = np.maximum(dots, 0)**2 / norm
    full = np.ones(y.size, dtype=bool)
    starts = []
    candidates = []
    for i in np.argsort(-merit):
        if any(np.linalg.norm(xy[i] - s) < rays.radius for s in starts):
            continue
        starts.append(xy[i])
        amp = max(dots[i] / norm[i], .001)
        p = np.r_[xy[i], amp, _linear(a, y - amp * paths[:, i], weights)]
        p = refine(p, y, full, .38)
        # Large skips exert no pull while choosing the physical basin.
        keep = np.abs(y - prediction(p)) < 2.
        p = refine(p, y, keep)
        e = y - prediction(p)
        candidates.append((np.minimum(e*e, 4.).sum(), p))
        if len(candidates) == 6:
            break
    p = min(candidates, key=lambda item: item[0])[1]
    return rays, p, y - prediction(p)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate plume center (m) and zero-sum receiver offsets (s)."""
    rays, p, _ = _fit(survey)
    clocks = rays.clocks @ p[5:]
    if np.max(np.abs(clocks)) > 4.:
        shift = brentq(lambda c: np.clip(clocks - c, -4., 4.).sum(),
                       clocks.min() - 4., clocks.max() + 4.)
        clocks = np.clip(clocks - shift, -4., 4.)
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clocks / 1000).tolist()}
