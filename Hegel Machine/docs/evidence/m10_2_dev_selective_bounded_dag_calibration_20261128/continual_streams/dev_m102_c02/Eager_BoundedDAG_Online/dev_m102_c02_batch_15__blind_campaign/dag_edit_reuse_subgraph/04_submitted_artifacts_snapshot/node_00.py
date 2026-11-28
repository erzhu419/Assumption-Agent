"""Invert the coupled sensor response and recover the reference coordinates."""
import numpy as np

_PAIR = np.array([2, 3, 0, 1])


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _sensor_parameters(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    adjusted = v - 0.18 * np.tanh(1.8 * u) + 0.32 * np.sin(2.4 * u[:, _PAIR])
    offsets, gains = np.empty(u.shape[1]), np.empty(u.shape[1])
    i, j = np.triu_indices(len(u), 1)
    for k, (a, b) in enumerate(zip(u.T, adjusted.T)):
        delta = a[j] - a[i]
        separated = np.abs(delta) > 0.1 * np.ptp(a)
        slope = np.median((b[j[separated]] - b[i[separated]]) / delta[separated])
        offset = np.median(b - slope * a)
        good = np.abs(b - (offset + slope * a)) < 0.5
        offsets[k], gains[k] = np.linalg.lstsq(
            _affine(a[good, None]), b[good], rcond=None
        )[0]
    return offsets, gains


def _invert(x, offset, gain):
    x = np.asarray(x, dtype=float)
    z = (x - offset) / gain
    axes = np.arange(z.shape[1])
    jac = np.zeros((len(z), len(axes), len(axes)))
    for _ in range(40):
        h = np.tanh(1.8 * z)
        phase = 2.4 * z[:, _PAIR]
        residual = offset + gain * z + 0.18 * h - 0.32 * np.sin(phase) - x
        if not np.any(np.abs(residual) > 1e-12):
            break
        jac[:, axes, axes] = gain + 0.324 * (1.0 - h * h)
        jac[:, axes, _PAIR] = -0.768 * np.cos(phase)
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= np.clip(step, -0.28, 0.28)
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    offset, gain = _sensor_parameters(study['panel_a'])
    panel = study['panel_b']

    def design(z):
        return _affine(np.cos(2.6 * z))

    operator = np.linalg.lstsq(design(np.asarray(panel['u'], dtype=float)),
                               np.asarray(panel['v'], dtype=float), rcond=None)[0]
    return {'c': design(_invert(study['train']['x'], offset, gain)) @ operator,
            'query': design(_invert(study['test']['x'], offset, gain)) @ operator}
