"""Recover latent coordinates and evaluate the two calibrated operators."""
import numpy as np


def _one(a):
    return np.column_stack((np.ones(len(a)), a))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _coupling(z):
    return .18 * np.tanh(1.8 * z) - .32 * np.sin(2.4 * np.roll(z, 2, axis=1))


def _calibration(panel):
    u = np.asarray(panel['u'], dtype=float)
    adjusted = np.asarray(panel['v'], dtype=float) - _coupling(u)
    i, j = np.triu_indices(len(u), 1)
    offset = np.empty(u.shape[1])
    slope = np.empty_like(offset)
    for k, (a, b) in enumerate(zip(u.T, adjusted.T)):
        da = a[i] - a[j]
        good = np.abs(da) > 1e-8
        m = np.median((b[i] - b[j])[good] / da[good])
        c = np.median(b - m * a)
        residual = np.abs(b - c - m * a)
        keep = residual <= max(.0005, 6 * np.median(residual))
        offset[k], slope[k] = _ls(_one(a[keep]), b[keep])
    return offset, slope


def _coords(panel, x):
    offset, slope = _calibration(panel)
    z = (x - offset) / slope
    target = x - offset
    # The diagonal derivative lies in [slope, slope + .324]. The only
    # off-diagonal derivative is bounded by .768. This iteration is a
    # global contraction for the calibrated slopes, even beyond the panel.
    step = 1 / (slope + .324)
    for _ in range(700):
        residual = slope * z + _coupling(z) - target
        if np.max(np.abs(residual), initial=0.) < 1e-11:
            break
        z -= step * residual
    return z


def _panels(panel, z):
    u = panel['u']

    def cyclic(q):
        return _one(q * np.roll(q, -1, axis=1))

    def cosine(q):
        return _one(np.cos(2.4 * q))

    return (cyclic(z) @ _ls(cyclic(u), panel['v1']),
            cosine(z) @ _ls(cosine(u), panel['v2']))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _coords(study['panel_a'], x)
    w = np.column_stack(_panels(study['panel_c'], z))
    return {'features': np.column_stack((z, w)), 'calibrated': w}
