"""Invert a borehole voltage survey with separable contact and anomaly terms."""

import numpy as np
from scipy.linalg import helmert
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize


class _Profile:
    """Least-squares profile with nonnegative anomaly amplitude."""

    def __init__(self, design, voltage, keep, tx, rx, length, depth, radius, weights=None):
        self.design, self.voltage = design[keep], voltage[keep]
        self.q = np.linalg.qr(self.design, mode='reduced')[0]
        self.y = self.project(self.voltage)
        self.tx, self.rx = tx[keep], rx[keep]
        self.length, self.depth = length, depth
        self.radius2 = radius**2
        self.weights = 1. if weights is None else weights[keep]
        self.scale = max(self.y @ self.y, 1e-20)

    def project(self, value):
        return value - np.einsum('ij,j->i', self.q, np.einsum('ij,i->j', self.q, value))

    def kernel(self, x, z):
        t = z - self.tx
        r = z - self.rx
        a = x*x + t*t + self.radius2
        b = (self.length-x)**2 + r*r + self.radius2
        return self.weights / np.sqrt(a*b), a, b, t, r

    def objective(self, u):
        x, z = u * [self.length, self.depth]
        k, a, b, t, r = self.kernel(x, z)
        p = self.project(k)
        amp = max(0., p @ self.y) / max(p @ p, 1e-30)
        e = self.y - amp*p
        dx = k*((self.length-x)/b - x/a)
        dz = -k*(t/a + r/b)
        g = (-2*amp/self.scale) * np.array([self.length*(e @ dx), self.depth*(e @ dz)])
        return (e @ e)/self.scale, g

    def search(self, radius, previous=None):
        nx = int(np.clip(np.ceil(2*self.length/radius)+1, 19, 65))
        nz = int(np.clip(np.ceil(2*self.depth/radius)+1, 25, 97))
        gx, gz = np.linspace(0., 1., nx), np.linspace(0., 1., nz)
        xx, zz = np.meshgrid(gx, gz, indexing='ij')
        uu = np.column_stack((xx.ravel(), zz.ravel()))
        x = self.length*uu[:, 0, None]
        z = self.depth*uu[:, 1, None]
        k = self.kernel(x, z)[0]
        # Compute the projected norms without forming a second full grid matrix.
        qk = np.einsum('ij,jk->ik', k, self.q)
        ky = np.einsum('ij,j->i', k, self.y)
        norm2 = np.einsum('ij,ij->i', k, k) - np.einsum('ij,ij->i', qk, qk)
        score = np.maximum(ky, 0.)**2 / np.maximum(norm2, 1e-30)
        surface = -score.reshape(nx, nz)
        local = np.flatnonzero((surface == minimum_filter(surface, size=3)).ravel())
        local = local[np.argsort(surface.ravel()[local])[:10]]
        starts = list(uu[local])
        if previous is not None:
            starts.append(previous)
        best = None
        for start in starts:
            fit = minimize(self.objective, start, jac=True, method='L-BFGS-B',
                           bounds=[(0., 1.), (0., 1.)],
                           options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 100})
            if best is None or fit.fun < best.fun:
                best = fit
        return best.x


def invert_ert_anomaly(survey: dict) -> dict:
    """Return the anomaly coordinates and zero-sum receiver contact voltages."""
    length, depth = float(survey['L_m']), float(survey['D_m'])
    radius = float(survey.get('a0_m', 6.))
    tx = np.asarray(survey['z_tx_m'], dtype=float)
    rx = np.asarray(survey['z_rx_m'], dtype=float)
    voltage = np.asarray(survey['V_obs'], dtype=float).ravel()
    ti = np.repeat(np.arange(tx.size), rx.size)
    ri = np.tile(np.arange(rx.size), tx.size)
    tb, rb = helmert(tx.size).T, helmert(rx.size).T
    t, r = tx[ti], rx[ri]
    d = r - t
    R = np.hypot(length, d)
    design = np.column_stack((length/R, (d/R)**2*(length/R), tb[ti], rb[ri]))

    # A nuisance-only fit supplies the initial mask; every channel is tested
    # again against the complete model, including the recovered anomaly.
    threshold = 0.0275
    coef = np.linalg.lstsq(design, voltage, rcond=None)[0]
    keep = np.abs(voltage - design @ coef) < threshold
    u = None
    for _ in range(6):
        profile = _Profile(design, voltage, keep, t, r, length, depth, radius)
        u = profile.search(radius, u)
        x, z = u * [length, depth]
        k = 1./np.sqrt((x*x + (z-t)**2 + radius**2) *
                       ((length-x)**2 + (z-r)**2 + radius**2))
        p = profile.project(k[keep])
        amp = max(0., p @ profile.y) / max(p @ p, 1e-30)
        coef = np.linalg.lstsq(design[keep], (voltage - amp*k)[keep], rcond=None)[0]
        new_keep = np.abs(voltage - design @ coef - amp*k) < threshold
        if np.array_equal(keep, new_keep):
            break
        if _ < 5:
            keep = new_keep
    # Correct only when the flagged channels, as a group, are consistent with
    # the specified discrete voltage step at the fitted noise level.
    residual = voltage - design @ coef - amp*k
    spikes = ~keep
    dof = max(1, keep.sum()-design.shape[1]-3)
    sigma = np.linalg.norm(residual[keep]) / np.sqrt(dof)
    step = 0.06 * np.sign(residual[spikes])
    discrepancy = residual[spikes] - step
    if spikes.any() and np.linalg.norm(discrepancy) <= 2.5*sigma*np.sqrt(spikes.sum()):
        corrected = voltage.copy()
        corrected[spikes] -= step
        # Partial pooling limits the impact of small departures from the step.
        w = np.where(spikes, np.sqrt(0.5), 1.)
        all_keep = np.ones_like(keep)
        profile = _Profile(design*w[:, None], corrected*w, all_keep, t, r, length, depth, radius, w)
        u = profile.search(radius, u)
        x, z = u * [length, depth]
        kw = profile.kernel(x, z)[0]
        p = profile.project(kw)
        amp = max(0., p @ profile.y) / max(p @ p, 1e-30)
        coef = np.linalg.lstsq(design*w[:, None], corrected*w - amp*kw, rcond=None)[0]
    offsets = rb @ coef[tx.size+1:]
    return {'anomaly_xz_m': [float(x), float(z)],
            'rx_contact_offset_v': offsets.tolist()}
