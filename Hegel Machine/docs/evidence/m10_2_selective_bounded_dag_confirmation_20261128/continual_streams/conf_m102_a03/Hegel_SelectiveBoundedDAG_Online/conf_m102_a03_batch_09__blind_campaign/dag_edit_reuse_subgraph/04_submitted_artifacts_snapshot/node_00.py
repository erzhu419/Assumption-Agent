"""Calibrate the sensor coordinates and construct the frozen DAG's features."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    # Exclude grossly corrupted reference rows before profiling the frequency.
    good = np.max(np.abs(v - u), axis=1) < 2.0
    u, v = u[good], v[good]
    skip = np.roll(u, -2, axis=1)
    bases = [np.column_stack((np.ones(len(u)), x, x**3)) for x in u.T]

    def profile(freq):
        designs = [np.column_stack((a, np.sin(freq * z)))
                   for a, z in zip(bases, skip.T)]
        coefs = [_fit(a, y) for a, y in zip(designs, v.T)]
        error = sum(np.sum((a @ c - y)**2)
                    for a, c, y in zip(designs, coefs, v.T))
        return error, np.column_stack(coefs)

    freq = minimize_scalar(lambda f: profile(f)[0], bounds=(1.5, 3.3),
                           method='bounded').x
    return freq, profile(freq)[1]


def _inverse(x, freq, coef):
    offset, gain, cubic, wave = coef
    idx = np.arange(len(gain))
    skip = (idx + 2) % len(idx)

    def forward(z):
        return offset + gain*z + cubic*z**3 + wave*np.sin(freq*z[skip])

    def jac(z):
        j = np.diag(gain + 3*cubic*z**2)
        j[idx, skip] += freq*wave*np.cos(freq*z[skip])
        return j

    # The locally linear inverse provides a start on the calibrated branch.
    start = np.linalg.solve(jac(np.zeros(len(gain))), (x-offset).T).T
    out = np.empty_like(start)
    for i, (row, initial) in enumerate(zip(x, start)):
        objective = lambda z: forward(z)-row
        result = least_squares(objective, np.clip(initial, -1.45, 1.45),
                               jac=jac, bounds=(-1.8, 1.8), gtol=1e-12)
        if np.linalg.norm(result.fun) > 1e-7:
            fallback = least_squares(objective, np.clip((row-offset)/gain, -1.45, 1.45),
                                     jac=jac, bounds=(-1.8, 1.8), gtol=1e-12)
            if fallback.cost < result.cost:
                result = fallback
        out[i] = result.x
    return out


def _readout(z):
    return np.column_stack((z, z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    freq, coef = _calibration(study['panel_a'])
    b = study['panel_b']
    transfer = _fit(_affine(b['u']), b['v'])
    x = np.asarray(study['train']['x'], dtype=float)
    xt = np.asarray(study['test']['x'], dtype=float)
    encoded = _affine(_inverse(np.concatenate((x, xt)), freq, coef)) @ transfer
    z, zt = encoded[:len(x)], encoded[len(x):]
    return dict(z=z, zt=zt, readout_train=_readout(z), readout_test=_readout(zt))
