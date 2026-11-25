"""Straight-ray inversion of cross-well first arrivals."""
import numpy as np
from scipy.linalg import helmert, qr
from scipy.optimize import brentq, least_squares, minimize, LinearConstraint, Bounds
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


def _fit_once(survey):
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
    e = y - prediction(p)
    clean = abs(e) < 2.
    col = np.column_stack((np.zeros((rcv.size, 5)), rays.clocks))

    def bounded(start, target, mask):
        fit = refine(start, target, mask)
        if np.max(abs(col @ fit)) <= 4.:
            return fit
        # Preserve the zero-sum gauge while fitting the active clock bounds.
        scale = 1. / np.maximum(np.linalg.norm(jac(fit)[mask], axis=0), 1e-7)
        def objective(v):
            b = fit + scale * v
            r = (prediction(b) - target)[mask]
            return 0.5 * (r @ r), scale * (jac(b)[mask].T @ r)
        shift = col @ fit
        result = minimize(objective, np.zeros(len(fit)), jac=True, method='SLSQP',
                          bounds=Bounds((lo-fit)/scale, (hi-fit)/scale),
                          constraints=LinearConstraint(col*scale, -4.-shift, 4.-shift),
                          options={'ftol': 1e-12, 'maxiter': 100})
        return fit + scale * result.x if result.success else fit

    p = bounded(p, y, clean)
    # A common cycle duration permits recovery of otherwise discarded picks.
    # Require clustering within ordinary pick noise before reusing them.
    e = y - prediction(p)
    skip = ~clean
    if skip.sum() >= 4:
        period = np.median(abs(e[skip]))
        noise = max(.01, 1.4826*np.median(abs(e[clean]-np.median(e[clean]))))
        if 4.5 < period < 7.5 and np.max(abs(abs(e[skip])-period)) < 4.*noise:
            target = y.copy()
            target[skip] -= period * np.sign(e[skip])
            p = bounded(p, target, full)
    return rays, p, y - prediction(p)


def _fit(survey):
    best = _fit_once(survey)
    rays, p, residual = best
    times = np.asarray(survey['T_obs_s'], dtype=float)
    nr, nc = times.shape
    y = (1000. * times).ravel()
    a = rays.design
    def prediction(b):
        return a @ b[3:] + b[2] * rays.plume(*b[:2])
    # Adjacent source differences cancel clocks. Robustly fitting the two
    # background columns here also avoids letting a majority skip band set
    # the clock or warp the background used to detect that band.
    dy = np.diff(1000. * times, axis=0).ravel()
    da = np.diff(a[:, :2].reshape(nr, nc, 2), axis=0).reshape(-1, 2)
    coef = p[3:5]
    for _ in range(8):
        weight = (1. + ((dy - da @ coef) / .18)**2)**(-.25)
        coef = _linear(da, dy, weight)
    smooth = y - a[:, :2] @ coef
    def split(column, gap):
        order = np.argsort(column)
        gaps = np.diff(column[order])
        if len(gaps) < 5:
            return None
        k = 3 + np.argmax(gaps[2:-2])
        if gaps[k-1] < gap:
            return None
        low, high = order[:k], order[k:]
        period = np.median(column[high]) - np.median(column[low])
        return (low, high, period) if 4.3 < period < 7.7 else None
    def score(b, e):
        ae = abs(e)
        skip = ae > 3.
        if skip.any():
            ae[skip] -= np.clip(np.median(ae[skip]), 4.5, 7.5)
        excess = np.maximum(abs(rays.clocks @ b[5:]) - 4., 0.)
        # Count a skip only after accounting for its waveform-independent
        # cycle duration. The measured clock bound also breaks gauge ties.
        return np.minimum(ae**2, 4.).sum() + .0001 * skip.sum() + nr * (excess @ excess)
    best_cost = score(p, residual)
    for j in range(nc):
        if split(smooth.reshape(nr, nc)[:, j], 2.6) is None:
            continue
        keep = np.arange(nc) != j
        _, ref, _ = _fit_once(dict(survey, z_rcv_m=np.asarray(survey['z_rcv_m'])[keep], T_obs_s=times[:, keep]))
        without_clock = y - a[:, :2] @ ref[3:5] - ref[2] * rays.plume(*ref[:2])
        band = split(without_clock.reshape(nr, nc)[:, j], 3.3)
        if band is None:
            continue
        low, high, period = band
        for rows, sign in ((low, -1.), (high, 1.)):
            indices = rows * nc + j
            corrected = y.copy()
            corrected[indices] -= sign * period
            _, trial, _ = _fit_once(dict(survey, T_obs_s=corrected.reshape(times.shape)/1000.))
            error = y - prediction(trial)
            skips = np.where(abs(error) > 3., np.sign(error), 0.)
            skips[indices] = sign
            lo = np.r_[0., 0., 0., np.full(a.shape[1], -np.inf), 4.5]
            hi = np.r_[float(survey['W_m']), float(survey['H_m']), np.full(a.shape[1]+1, np.inf), 7.5]
            def jac(b):
                g, gx, gz = rays.plume(*b[:2], derivative=True)
                return np.column_stack((b[2]*gx, b[2]*gz, g, a, skips))
            fit = least_squares(lambda b: prediction(b[:-1]) + skips*b[-1] - y,
                                np.r_[trial, np.clip(period, 4.5, 7.5)], jac=jac,
                                bounds=(lo, hi), x_scale='jac', loss='soft_l1',
                                f_scale=.15, max_nfev=50).x[:-1]
            error = y - prediction(fit)
            cost = score(fit, error)
            if cost < best_cost:
                best_cost = cost
                best = rays, fit, error
    return best


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate plume center (m) and zero-sum receiver offsets (s)."""
    rays, p, _ = _fit(survey)
    clocks = rays.clocks @ p[5:]
    if np.max(np.abs(clocks)) > 4.:
        shift = brentq(lambda c: np.clip(clocks - c, -4., 4.).sum(),
                       clocks.min() - 4., clocks.max() + 4.)
        clocks = np.clip(clocks - shift, -4., 4.)
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clocks / 1000).tolist()}
