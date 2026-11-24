"""Invert the downstream gauge records with the specified finite-volume model."""

import numpy as np
from scipy.optimize import minimize_scalar


class _Canal:
    def __init__(self, bundle):
        self.n = bundle['N']
        self.steps = bundle['T_steps']
        self.ratio = bundle['dt'] / bundle['dx']
        x = (np.arange(self.n) + 0.5) * bundle['dx']
        self.bed_shape = np.exp(-0.5 * ((x - 400.0) / 45.0)**2)
        self.bay = x < 80.0
        self.gauges = np.asarray(bundle['gauge_cells'], dtype=int)
        self.indices = np.asarray(bundle['obs_step_indices'], dtype=int)
        self.sample_rows = {t: k for k, t in enumerate(self.indices)}

    def simulate(self, heights, modes, full=False):
        heights = np.atleast_1d(heights)
        modes = np.broadcast_to(modes, heights.shape)
        bed = heights[:, None] * self.bed_shape
        face_bed = np.maximum(np.pad(bed, ((0, 0), (1, 0)), mode='edge'),
                              np.pad(bed, ((0, 0), (0, 1)), mode='edge'))
        eta = np.full_like(bed, 2.0)
        eta[:, self.bay] = np.where(modes, 2.65, 1.45)[:, None]
        h = eta - bed
        q = np.zeros_like(h)
        stop = self.steps if full else self.indices.max() + 1
        shape = (self.steps, self.n) if full else (len(self.indices), len(self.gauges))
        out = np.empty((len(heights), *shape))
        for t in range(stop):
            if full:
                out[:, t] = eta
            elif t in self.sample_rows:
                out[:, self.sample_rows[t]] = eta[:, self.gauges]
            if t == stop - 1:
                break
            uv = q / np.maximum(h, 1e-9)
            ep = np.concatenate((eta[:, :1], eta, eta[:, -1:]), axis=1)
            up = np.concatenate((-uv[:, :1], uv, -uv[:, -1:]), axis=1)
            hl = np.maximum(ep[:, :-1] - face_bed, 0.0)
            hr = np.maximum(ep[:, 1:] - face_bed, 0.0)
            ul, ur = up[:, :-1], up[:, 1:]
            ql, qr = hl * ul, hr * ur
            speed = np.maximum(np.maximum(np.abs(ul) + np.sqrt(9.81 * hl),
                                          np.abs(ur) + np.sqrt(9.81 * hr)), 1e-6)
            fh = 0.5 * (ql + qr - speed * (hr - hl))
            fl = ql * ul + 4.905 * hl**2
            fr = qr * ur + 4.905 * hr**2
            fq = 0.5 * (fl + fr - speed * (qr - ql))
            dq_bed = 4.905 * (hl[:, 1:]**2 - hr[:, :-1]**2)
            h -= self.ratio * np.diff(fh, axis=1)
            q += self.ratio * (dq_bed - np.diff(fq, axis=1))
            eta = h + bed
        return out


def invert_canal_regime(obs_bundle: dict) -> dict:
    """Fit sill height and release mode, then replay the entire canal."""
    model = _Canal(obs_bundle)
    observations = np.asarray(obs_bundle['eta_obs'], dtype=float)
    grid = np.linspace(0.0, 0.75, 13)
    heights = np.tile(grid, 2)
    modes = np.repeat([False, True], len(grid))
    residual = model.simulate(heights, modes) - observations
    scores = np.mean(residual**2, axis=(1, 2))
    best = (np.inf, 0.0, False)
    for mode, row in zip((False, True), scores.reshape(2, -1)):
        def objective(height):
            error = model.simulate(height, mode)[0] - observations
            return np.mean(error**2)

        k = int(np.argmin(row))
        if row[k] < best[0]:
            best = (row[k], grid[k], mode)
        result = minimize_scalar(objective, method='bounded',
                                 bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                                 options={'xatol': 2e-6})
        if result.fun < best[0]:
            best = (result.fun, result.x, mode)
    _, height, mode = best
    return {'sluice_high_mode': mode, 'sill_height_m': float(height),
            'eta_full': model.simulate(height, mode, full=True)[0]}
