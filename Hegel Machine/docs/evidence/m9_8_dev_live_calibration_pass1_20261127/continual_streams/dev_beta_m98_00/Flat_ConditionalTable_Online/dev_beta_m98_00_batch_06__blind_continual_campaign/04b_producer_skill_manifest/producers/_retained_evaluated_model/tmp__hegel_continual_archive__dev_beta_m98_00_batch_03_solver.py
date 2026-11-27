"""Finite-volume inversion for the periodic chamber campaign."""

from itertools import product

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibration(sweeps):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    design = np.column_stack((np.ones(len(ref)), ref))
    bank = sweeps['single_bank_sweeps']
    bank_id = np.asarray(bank['bank_id'])
    signal = np.asarray(bank['ref_signal'])
    readout = np.asarray(bank['raw_readout'])
    coeff = np.empty((5, 4))
    for channel in range(4):
        select = bank_id == channel
        single = np.zeros((select.sum(), 5))
        single[:, 0] = 1
        single[:, channel + 1] = signal[select]
        coeff[:, channel] = np.linalg.lstsq(
            np.vstack((design, single)), np.r_[raw[:, channel], readout[select]], rcond=None
        )[0]

    def calibrate(values):
        rows = np.asarray(values).reshape(-1, 4)
        return np.linalg.solve(coeff[1:].T, (rows - coeff[0]).T).T.ravel()

    return calibrate


def _axis(kappa, velocity, h, robin, boundary):
    k, v = np.asarray(kappa), np.asarray(velocity)
    face = 0.5 * (k[:-1] + k[1:]) / h**2
    left = -np.r_[face[:-1] + np.maximum(v[1:-1], 0) / h, k[-1] / h**2]
    right = -np.r_[k[0] / h**2, face[1:] + np.maximum(-v[1:-1], 0) / h]
    diag = np.r_[-right[0], (face[:-1] + face[1:]) + np.abs(v[1:-1]) / h, -left[-1]]
    if boundary == 'periodic':
        seam = 0.5 * (k[0] + k[-1]) / h**2
        right[0] = -face[0] - max(-v[0], 0) / h
        left[-1] = -face[-1] - max(v[-1], 0) / h
        across_low = -seam - max(v[0], 0) / h
        across_high = -seam - max(-v[-1], 0) / h
        diag[0] = -right[0] - across_low
        diag[-1] = -left[-1] - across_high
        return diags((left, diag, right, [across_high], [across_low]),
                     (-1, 0, 1, 1 - len(k), len(k) - 1), format='csc')
    if boundary == 'open':
        diag[[0, -1]] += robin / h
    return diags((left, diag, right), (-1, 0, 1), format='csc')


def _response(study, boundaries, modes):
    nx, ny = study['nx'], study['ny']
    axes = [_axis(study['kappa_' + d], study['vel_' + d], study['L' + d] / (n - 1),
                  study['robin_' + d], boundary)
            for d, n, boundary in zip(('x', 'y'), (nx, ny), boundaries)]
    op = (kron(eye(ny), axes[0]) + kron(axes[1], eye(nx))
          + diags(np.asarray(study['V_reaction_2d']).ravel(order='F'))).tocsc()
    ident = eye(nx * ny, format='csc')
    rhs = np.asarray(modes).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = np.asarray(study['sensor_ij'])
    sensors = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        field = splu(op + shift * ident).solve(rhs)
        blocks.append(field[sensors])
    return np.vstack(blocks), field.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _boundaries(study, calibrate, sources):
    pulses = study.get('reference_tracer_pulses')
    default = ('periodic', 'periodic')
    if not pulses:
        return default
    weights = np.stack([p['w_pulse'] for p in pulses])
    modes = np.tensordot(weights, sources, axes=1)
    observed = np.column_stack([calibrate(p['y_pulse_raw']) for p in pulses])
    candidates = product(('periodic', 'open', 'sealed'), repeat=2)
    return min(candidates, key=lambda pair: np.linalg.norm(_response(study, pair, modes)[0] - observed))

def _project(q, cap, total):
    if total <= 0:
        return np.zeros_like(q)
    if total >= len(q) * cap:
        return np.full_like(q, cap)
    lo, hi = np.min(q) - cap, np.max(q)
    for _ in range(53):
        mid = (lo + hi) * .5
        if np.clip(q - mid, 0., cap).sum() > total:
            lo = mid
        else:
            hi = mid
    return np.clip(q - (lo + hi) * .5, 0., cap)


def _dual(g, w, cap):
    free = (w > 1e-10) & (w < cap - 1e-10)
    if free.any():
        return float(np.mean(g[free]))
    upper = w > cap / 2
    if upper.all():
        return float(g.max())
    if not upper.any():
        return float(g.min())
    return float(.5 * (g[upper].max() + g[~upper].min()))


def _weights(s, y, ridge, cap, total):
    h = s.T @ s + ridge * np.eye(s.shape[1])
    b = s.T @ y
    w = np.full(len(b), total / len(b))
    eig = np.linalg.eigvalsh(h)
    lip = eig[-1]
    if lip == 0 or total <= 0 or total >= len(b) * cap:
        return w
    beta = (np.sqrt(lip) - np.sqrt(max(eig[0], 0.))) / (np.sqrt(lip) + np.sqrt(max(eig[0], 0.)))
    q = w.copy()
    for i in range(2000):
        nxt = _project(q - (h @ q - b) / lip, cap, total)
        q = nxt + beta * (nxt - w)
        w = nxt
        g = h @ w - b
        d = g - _dual(g, w, cap)
        gap = np.maximum(d, 0.) @ w + np.maximum(-d, 0.) @ (cap - w)
        if gap < 1e-12:
            break
        if i % 8 == 0:
            # Solve on the current face, accepting only a feasible KKT point.
            free = (w > 1e-10) & (w < cap - 1e-10)
            if free.any():
                trial = np.where(w >= cap - 1e-10, cap, 0.)
                hf = h[np.ix_(free, free)]
                ones = np.ones(free.sum())
                try:
                    sol, direction = np.linalg.solve(hf, np.column_stack((b[free] - h[free] @ trial, ones))).T
                except np.linalg.LinAlgError:
                    continue
                nu = (total - trial.sum() - sol.sum()) / direction.sum()
                trial[free] = sol + nu * direction
                d = h @ trial - b - nu
                if (trial.min() >= -1e-12 and trial.max() <= cap + 1e-12 and
                    np.min(d[trial <= 0], initial=0.) >= -1e-11 and
                    np.max(d[trial >= cap], initial=0.) <= 1e-11):
                    return np.clip(trial, 0., cap)
    return w


def solve_batch(study: dict) -> dict:
    source = np.asarray(study['source_modes'])
    calibrate = _calibration(study['sensor_calibration_sweeps'])
    wall = _boundaries(study, calibrate, source)
    response, fields = _response(study, wall, source)
    observed = calibrate(study['y_obs_raw'])
    z = np.asarray(study['Z_nuisance'])
    pz = np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    s = response - z @ (pz @ response)
    y = observed - z @ (pz @ observed)
    ridge, box, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w = _weights(s, y, ridge, box, budget)
    residual = s @ w - y
    obj = float((residual @ residual + ridge * (w @ w)) / 2)
    g = s.T @ residual + ridge * w
    nu = _dual(g, w, box)
    lower = np.maximum(0., g - nu)
    upper = np.maximum(0., nu - g)
    dual = obj - float(g @ w) + nu * budget - box * float(upper.sum())
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': obj, 'dual_lower_bound': dual}}
