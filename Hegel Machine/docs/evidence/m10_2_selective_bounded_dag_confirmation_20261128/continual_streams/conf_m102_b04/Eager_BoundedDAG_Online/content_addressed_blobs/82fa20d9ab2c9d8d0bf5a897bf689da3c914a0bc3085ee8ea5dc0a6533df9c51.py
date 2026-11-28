"""Separate Fourier heads for a calibrated scalar response and trajectory."""
from itertools import combinations

import numpy as np


def _calibrate(panel, x):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    # The panel describes independent affine gain and offset calibrations.
    gain, offset = [], []
    for a, b in zip(u.T, v.T):
        da = a[:, None] - a
        db = b[:, None] - b
        ids = np.triu(np.abs(da) > 1e-7, 1)
        g = np.median(db[ids] / da[ids])
        c = np.median(b - g * a)
        e = np.abs(b - (g * a + c))
        keep = e <= max(1e-8, 6 * np.median(e))
        if keep.sum() >= 3:
            c, g = np.linalg.lstsq(np.column_stack((np.ones(keep.sum()), a[keep])), b[keep], rcond=None)[0]
        gain.append(g)
        offset.append(c)
    gain = np.asarray(gain)
    return (x - offset) / np.where(np.abs(gain) > 1e-8, gain, 1.0)


def _harmonics(x, w):
    return np.column_stack((np.sin(w*x), np.cos(w*x)))


def _scalar_basis(x):
    # Even harmonics and their pair products preserve reflection symmetry
    # around the calibrated origin, including the diagonal (square) terms.
    c = np.cos(2.6*x)
    pairs = [c[:, i]*c[:, j] for i, j in combinations(range(c.shape[1]), 2)]
    return np.column_stack((np.ones(len(c)), c, *pairs, c*c))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    frozen = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    hybrid = np.asarray(frozen['hybrid'], dtype=float)
    d = x.shape[1]
    # The final basis block begins with 1 and the frozen calibrated coordinates.
    k = (d + 1)*(d + 2)*(d + 3)//6
    upstream_x = hybrid[:, -k+1:-k+1+d]
    # Prefer the upstream coordinates when the independent panel fit confirms
    # them; fall back to the robust diagonal fit if the panel disagrees.
    if np.allclose(upstream_x, x, rtol=1e-6, atol=1e-7):
        x = upstream_x
    f = _scalar_basis(x)
    coef = np.linalg.lstsq(f[:n], np.asarray(train['y'], dtype=float), rcond=1e-10)[0]
    y = f[n:] @ coef
    z = np.asarray(train['traj'], dtype=float)
    basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    rates = np.column_stack((np.ones(len(x)), x, _harmonics(x, 2.6)))
    amps = np.linalg.lstsq(rates[:n], z @ basis.T, rcond=None)[0]
    state = (rates[n:] @ amps) @ basis
    dv = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': dv}
