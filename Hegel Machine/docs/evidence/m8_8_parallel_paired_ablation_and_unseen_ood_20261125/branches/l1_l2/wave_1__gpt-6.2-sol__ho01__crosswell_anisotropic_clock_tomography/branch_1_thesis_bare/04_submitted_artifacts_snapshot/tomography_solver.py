"""Joint inversion of straight-ray traveltimes for a Gaussian plume and receiver statics."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


class _SurveyModel:
    def __init__(self, survey):
        self.w = float(survey['W_m'])
        self.h = float(survey['H_m'])
        self.r = float(survey['R_plume_m'])
        source = np.asarray(survey['z_src_m'], dtype=float)
        receiver = np.asarray(survey['z_rcv_m'], dtype=float)
        dz = (receiver[None, :] - source[:, None]).ravel()
        self.nrcv = nr = receiver.size
        self.a = np.repeat(source, nr)
        self.length = np.hypot(self.w, dz)
        self.cos = self.w / self.length
        self.sin = dz / self.length
        # Clock columns span the sum-zero subspace. The two background
        # coefficients are s0 and the product s0 * epsilon, in ms/m.
        clock = np.vstack((np.eye(nr - 1), -np.ones((1, nr - 1))))
        self.base = np.column_stack((self.length, -self.length * self.sin**2,
                                     np.tile(clock, (source.size, 1))))
        self.y = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000.0

    def plume(self, x, z, derivatives=False):
        """Exact Gaussian integral on each finite source-receiver segment."""
        u = x * self.cos + (z - self.a) * self.sin
        v = (z - self.a) * self.cos - x * self.sin
        t0 = u / (np.sqrt(2) * self.r)
        t1 = (self.length - u) / (np.sqrt(2) * self.r)
        e = np.exp(-0.5 * (v / self.r)**2)
        g = self.r * np.sqrt(np.pi / 2) * e * (erf(t0) + erf(t1))
        if not derivatives:
            return g
        gu = e * (np.exp(-t0*t0) - np.exp(-t1*t1))
        gv = -v * g / self.r**2
        return g, gu * self.cos - gv * self.sin, gu * self.sin + gv * self.cos


class _Profile:
    """Eliminate background and clocks from each weighted plume fit."""
    def __init__(self, model, y, weight):
        self.model = model
        self.w = np.sqrt(weight)
        self.design = self.w[:, None] * model.base
        self.q = np.linalg.qr(self.design, mode='reduced')[0]
        self.y = y
        self.target = self.project(y * self.w)

    def project(self, v):
        # Explicit contractions are efficient for the small nuisance subspace.
        coords = np.einsum('ni,...n->...i', self.q, v)
        return v - np.einsum('ni,...i->...n', self.q, coords)

    def refine(self, start):
        m = self.model

        def residual(p):
            return p[2] * self.project(self.w * m.plume(*p[:2])) - self.target

        def jacobian(p):
            g, gx, gz = m.plume(*p[:2], derivatives=True)
            return self.project(self.w * np.array([p[2]*gx, p[2]*gz, g])).T

        return least_squares(residual, start, jac=jacobian,
                             bounds=([0, 0, 0], [m.w, m.h, np.inf]),
                             x_scale=[m.r, m.r, 0.1], max_nfev=80)

    def locate(self):
        m = self.model
        step = m.r / 2
        xs = np.linspace(0, m.w, int(np.ceil(m.w / step)) + 1)
        zs = np.linspace(0, m.h, int(np.ceil(m.h / step)) + 1)
        x, z = (a.ravel() for a in np.meshgrid(xs, zs))
        candidates = []
        for i in range(0, x.size, 64):
            j = slice(i, i + 64)
            template = self.project(self.w * m.plume(x[j, None], z[j, None]))
            cross = np.maximum(0, np.einsum('ij,j->i', template, self.target))
            norm2 = np.einsum('ij,ij->i', template, template)
            candidates.extend(zip(cross**2 / norm2, x[j], z[j], cross / norm2))
        fits = []
        for _, x0, z0, amp in sorted(candidates, reverse=True):
            if all(np.hypot(x0 - x1, z0 - z1) > m.r for x1, z1, _ in fits):
                fits.append((x0, z0, self.refine([x0, z0, amp])))
                if len(fits) == 4:
                    break
        return min((fit for _, _, fit in fits), key=lambda fit: fit.cost).x

    def linear_terms(self, p):
        return np.linalg.lstsq(self.design,
                               self.w * (self.y - p[2] * self.model.plume(*p[:2])),
                               rcond=None)[0]


def _clock_projection(clock):
    """Project onto the box with zero mean, in milliseconds."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = min(clock) - 4.0, max(clock) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (hi + lo) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return the plume center in meters and the receiver biases in seconds."""
    m = _SurveyModel(survey)
    design, y = m.base, m.y
    # A robust background-only fit identifies the gross skips without letting
    # one such pick move an entire receiver's clock by a substantial fraction.
    weights = np.ones_like(y)
    for _ in range(12):
        w = np.sqrt(weights)
        beta = np.linalg.lstsq(w[:, None] * design, w * y, rcond=None)[0]
        residual = y - np.einsum('ij,j->i', design, beta)
        weights = 1 / np.maximum(1, np.abs(residual) / 0.35)

    prof = _Profile(m, y, np.abs(residual) < 2.75)
    p = prof.locate()
    for _ in range(3):
        beta = prof.linear_terms(p)
        residual = y - np.einsum('ij,j->i', design, beta) - p[2] * m.plume(*p[:2])
        inlier = np.abs(residual) < 2.75
        if np.array_equal(inlier, prof.w):
            break
        prof = _Profile(m, y, inlier)
        p = prof.refine(p).x

    # A coherent signed time jump can be fitted as a nuisance parameter,
    # recovering the useful information in skipped picks. If the outliers do
    # not share a common jump magnitude, keep the inlier-only fit instead.
    beta = prof.linear_terms(p)
    residual = y - np.einsum('ij,j->i', design, beta) - p[2] * m.plume(*p[:2])
    good = prof.w.astype(bool)
    skipped = ~good
    if np.count_nonzero(skipped) >= 4:
        jumps = np.abs(residual[skipped])
        offset = np.median(jumps)
        noise = max(0.015, 1.4826 * np.median(np.abs(residual[good])))
        coherent = np.abs(jumps - offset) < 3.5 * noise
        if np.count_nonzero(coherent) >= 0.75 * jumps.size:
            sign = np.zeros_like(y)
            indices = np.flatnonzero(skipped)[coherent]
            sign[indices] = np.sign(residual[indices])
            m.base = np.column_stack((design, sign))
            prof = _Profile(m, y, good | (sign != 0))
            p = prof.refine(p).x
            beta = prof.linear_terms(p)

    c = beta[2:1 + m.nrcv]
    clock = _clock_projection(np.r_[c, -c.sum()])
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clock / 1000).tolist()}
