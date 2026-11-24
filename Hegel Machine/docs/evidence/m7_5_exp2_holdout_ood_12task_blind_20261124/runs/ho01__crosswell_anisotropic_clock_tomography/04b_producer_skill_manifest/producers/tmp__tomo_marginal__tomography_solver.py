"""Straight-ray traveltime inversion for a Gaussian plume and receiver clocks."""
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.special import erf, log_ndtr


class _Rays:
    def __init__(self, w, zs, zr, radius):
        self.r = radius
        dz = zr[None, :] - zs[:, None]
        self.length = np.hypot(w, dz).ravel()
        self.u = w / self.length
        self.v = dz.ravel() / self.length
        self.zs = np.repeat(zs, len(zr))
        self.receiver = np.tile(np.arange(len(zr)), len(zs))
        # Last clock is minus the sum of the others. The two background
        # coefficients are s0 and s0*epsilon, expressed in ms/m.
        clocks = np.eye(len(zr), dtype=float)[self.receiver]
        self.design = np.column_stack((self.length, -self.length * self.v**2,
                                       clocks[:, :-1] - clocks[:, -1:]))

    def kernel(self, centers, ind, derivative=False):
        x, z = np.asarray(centers).T
        u, v, length = self.u[ind], self.v[ind], self.length[ind]
        t = z[..., None] - self.zs[ind]
        p = u * t - v * x[..., None]
        q = v * t + u * x[..., None]
        r = self.r
        a, b = q / (np.sqrt(2) * r), (length - q) / (np.sqrt(2) * r)
        normal = np.exp(-0.5 * (p / r)**2)
        g = r * np.sqrt(np.pi / 2) * normal * (erf(a) + erf(b))
        if not derivative:
            return g
        gp = -p / r**2 * g
        gq = normal * (np.exp(-a*a) - np.exp(-b*b))
        return g, np.column_stack((-v*gp + u*gq, u*gp + v*gq))


def _clip_clock(clock):
    """Project onto the zero-sum clock box, in milliseconds."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = np.min(clock)-4.0, np.max(clock)+4.0
    for _ in range(52):
        mid = (lo + hi) / 2
        if np.clip(clock-mid, -4., 4.).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock-(lo+hi)/2, -4., 4.)


def _invert(survey, include=None):
    w, h = float(survey['W_m']), float(survey['H_m'])
    zs = np.asarray(survey['z_src_m'], dtype=float)
    zr = np.asarray(survey['z_rcv_m'], dtype=float)
    rays = _Rays(w, zs, zr, float(survey.get('R_plume_m', 5.5)))
    y = 1000. * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    available = np.isfinite(y) if include is None else np.isfinite(y) & include.ravel()
    D = rays.design
    Da, ya = D[available], y[available]
    bg = np.linalg.lstsq(Da, ya, rcond=None)[0]
    for _ in range(12):
        resid = Da @ bg - ya
        weight = (1 + (resid / .25)**2)**(-.25)
        bg = np.linalg.lstsq(Da * weight[:, None], ya * weight, rcond=None)[0]
    good = available & (np.abs(D @ bg-y) < 2.7)

    step = rays.r / 2.
    X, Z = np.meshgrid(np.linspace(0, w, int(np.ceil(w/step))+1),
                       np.linspace(0, h, int(np.ceil(h/step))+1))
    grid = np.column_stack((X.ravel(), Z.ravel()))
    xy = None
    for _ in range(5):
        ind = np.flatnonzero(good)
        Q, U = qr(D[ind], mode='economic')
        coeff_y = Q.T @ y[ind]
        yy = y[ind] - Q @ coeff_y

        def project(g):
            return g - Q @ (Q.T @ g)

        if xy is None:
            best = -1.
            for off in range(0, len(grid), 128):
                block = grid[off:off+128]
                raw = rays.kernel(block, ind)
                q = np.einsum('ij,jk->ik', raw, Q)
                # Profile the positive amplitude and the linear nuisance terms.
                power = np.sum(raw*raw, axis=1) - np.sum(q*q, axis=1)
                dot = np.einsum('ij,j->i', raw, yy)
                score = np.maximum(dot, 0.)**2 / np.maximum(power, 1.e-20)
                winner = int(np.argmax(score))
                if score[winner] > best:
                    xy, best = block[winner], score[winner]

        def profile(pos, jac=False):
            if jac:
                raw, dg = rays.kernel(pos, ind, derivative=True)
                dk = project(dg)
            else:
                raw = rays.kernel(pos, ind)
            k = project(raw)
            denom = max(np.dot(k, k), 1.e-20)
            amp = max(0., np.dot(k, yy) / denom)
            if not jac:
                return amp*k - yy
            if amp == 0.:
                return np.zeros_like(dk)
            da = ((yy - 2*amp*k) @ dk) / denom
            return amp*dk + k[:, None]*da

        xy = least_squares(profile, xy, jac=lambda x: profile(x, True),
                           bounds=([0., 0.], [w, h]), max_nfev=50).x
        g = rays.kernel(xy, ind)
        k = project(g)
        amplitude = max(0., np.dot(k, yy) / max(np.dot(k, k), 1.e-20))
        coef = solve_triangular(U, coeff_y - amplitude * (Q.T @ g))
        prediction = D @ coef + amplitude * rays.kernel(xy, slice(None))
        updated = available & (np.abs(y-prediction) < 2.7)
        if np.array_equal(updated, good):
            break
        good = updated

    # Integrate over positive amplitude when the center likelihood is broad.
    e = (y - prediction)[ind]
    sig = max(1.e-9, 1.4826*np.median(np.abs(e-np.median(e))))
    sing = np.linalg.svd(profile(xy, True), compute_uv=False)
    if sig / max(sing[-1], 1.e-20) > .75:
        dx = rays.r / 10.
        xx = np.linspace(0., w, int(np.ceil(w/dx))+1)
        zz = np.linspace(0., h, int(np.ceil(h/dx))+1)
        gx, gz = np.meshgrid(xx, zz)
        centers = np.column_stack((gx.ravel(), gz.ravel()))
        alpha = (sig/.035)**2
        prior_mean = .065
        log_mass = []
        for off in range(0, len(centers), 256):
            raw = rays.kernel(centers[off:off+256], ind)
            q = np.einsum('ij,jk->ik', raw, Q)
            power = np.maximum(np.sum(raw*raw, axis=1)-np.sum(q*q, axis=1), 0.) + alpha
            dot = np.einsum('ij,j->i', raw, yy) + alpha*prior_mean
            t = dot/(sig*np.sqrt(power))
            log_mass.append(.5*t*t-.5*np.log(power)+log_ndtr(t))
        log_mass = np.concatenate(log_mass)
        mass = np.exp(log_mass-np.max(log_mass)).reshape(len(zz),len(xx))
        mass[[0,-1],:] *= .5
        mass[:,[0,-1]] *= .5
        xy = np.einsum('i,ij->j', mass.ravel(), centers)/mass.sum()
        g = rays.kernel(xy, ind)
        k = project(g)
        amplitude = max(0., np.dot(k, yy)/max(np.dot(k, k), 1.e-20))
        coef = solve_triangular(U, coeff_y-amplitude*(Q.T@g))
        prediction = D@coef + amplitude*rays.kernel(xy, slice(None))
    clock = _clip_clock(np.r_[coef[2:], -np.sum(coef[2:])])
    return xy, clock/1000., prediction/1000.


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in meters and zero-sum receiver biases in seconds."""
    xy, clock, _ = _invert(survey)
    return {'plume_xz_m': xy.tolist(), 'rcv_clock_bias_s': clock.tolist()}
