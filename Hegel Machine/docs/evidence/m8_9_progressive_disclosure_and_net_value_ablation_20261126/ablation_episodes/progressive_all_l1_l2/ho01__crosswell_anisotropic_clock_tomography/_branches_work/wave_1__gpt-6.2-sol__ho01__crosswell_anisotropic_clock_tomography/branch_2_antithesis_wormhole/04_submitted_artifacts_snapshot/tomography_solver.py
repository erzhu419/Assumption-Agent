"""Straight-ray inversion for one Gaussian plume and receiver clock offsets."""
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.special import ndtr


class _RayModel:
    def __init__(self, survey):
        self.w, self.h = float(survey['W_m']), float(survey['H_m'])
        self.r = float(survey.get('R_plume_m', 5.5))
        zs = np.asarray(survey['z_src_m'], dtype=float)
        zr = np.asarray(survey['z_rcv_m'], dtype=float)
        self.ns, self.nr = len(zs), len(zr)
        self.zs = np.repeat(zs, self.nr)
        dz = (zr[None, :] - zs[:, None]).ravel()
        length = np.hypot(self.w, dz)
        self.length = length
        self.u, self.v = self.w / length, dz / length
        self.clock = helmert(self.nr).T if self.nr > 1 else np.empty((1, 0))
        self.design = np.column_stack((length / self.w, -length * self.v**2 / self.w,
                                       np.tile(self.clock, (self.ns, 1))))
        self.y = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()

    def kernel(self, x, z, gradient=False):
        r = self.r
        dz = z - self.zs
        a = (self.u*x + self.v*dz) / r
        b = (self.length / r) - a
        d = (-self.v*x + self.u*dz) / r
        cross = np.exp(-0.5*d*d)
        k = np.sqrt(2*np.pi)*cross*(ndtr(b)-ndtr(-a))
        if not gradient:
            return k
        along = cross*(np.exp(-0.5*a*a)-np.exp(-0.5*b*b)) / r
        across = -k*d/r
        return k, along*self.u-across*self.v, along*self.v+across*self.u

    def fit(self, p, y, loss='soft_l1', scale=0.20, keep=slice(None)):
        design = self.design[keep]

        def fun(v):
            return design @ v[3:] + v[2]*self.kernel(*v[:2])[keep] - y

        def jac(v):
            k, kx, kz = self.kernel(*v[:2], gradient=True)
            return np.column_stack((v[2]*kx[keep], v[2]*kz[keep], k[keep], design))

        n = self.design.shape[1]
        return least_squares(fun, p, jac=jac, loss=loss, f_scale=scale,
                             bounds=(np.r_[0, 0, 0, np.full(n, -np.inf)],
                                     np.r_[self.w, self.h, np.full(n+1, np.inf)]),
                             x_scale='jac', max_nfev=100, ftol=1e-8)

    def starts(self):
        A, y = self.design, self.y
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(15):
            err = y - A @ beta
            w = (1 + (err / 0.35)**2)**(-0.25)
            beta = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
        q, _ = np.linalg.qr(A*w[:, None], mode='reduced')
        v = (y-A @ beta)*w
        v -= q @ (q.T @ v)
        step = self.r * 0.53
        gx = np.linspace(0, self.w, int(np.ceil(self.w / step)) + 1)
        gz = np.linspace(0, self.h, int(np.ceil(self.h / step)) + 1)
        points = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
        scores, strengths = [], []
        for x, z in points:
            wk = self.kernel(x, z)*w
            wk -= q @ (q.T @ wk)
            norm_sq = wk @ wk
            ip = wk @ v
            scores.append(max(ip, 0.) / np.sqrt(max(norm_sq, 1e-20)))
            strengths.append(max(ip / max(norm_sq, 1e-20), 0.02))
        chosen = []
        for i in np.argsort(scores)[::-1]:
            loc = points[i]
            if any(np.linalg.norm(loc - old[:2]) < 1.5*self.r for old in chosen):
                continue
            beta = np.linalg.lstsq(A*w[:, None], (y-strengths[i]*self.kernel(*loc))*w,
                                   rcond=None)[0]
            chosen.append(np.r_[loc, strengths[i], beta])
            if len(chosen) == 6:
                break
        return chosen

    def solve(self):
        fits = [self.fit(p, self.y) for p in self.starts()]
        best = min(fits, key=lambda f: f.cost)
        keep = np.abs(best.fun) < 2.5
        refined = self.fit(best.x, self.y[keep], loss='soft_l1', scale=0.40, keep=keep)
        return refined.x


def _bound_clocks(clock):
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = np.min(clock)-4, np.max(clock)+4
    for _ in range(48):
        mid = (lo+hi)/2
        if np.clip(clock-mid, -4, 4).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock-(lo+hi)/2, -4, 4)


def invert_plume_and_clocks(survey: dict) -> dict:
    model = _RayModel(survey)
    p = model.solve()
    clocks = _bound_clocks(model.clock @ p[5:]) / 1000
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': clocks.tolist()}
