"""Calibrate the four-channel load bridge in canonical load coordinates."""

import numpy as np
from scipy.optimize import least_squares


_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4
_PREV = (_CHANNEL - 1) % 4
_PAIRS = np.triu_indices(4)


def _fit_bridge(loads, raw):
    # Use a linear-in-parameters pilot to locate gross errors by whole row.
    a = np.column_stack((np.ones(len(loads)), loads,
                         loads[:, _PAIRS[0]] * loads[:, _PAIRS[1]], loads**3))
    weight = np.ones(len(loads))
    for _ in range(40):
        pilot = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
        error = np.linalg.norm(a @ pilot - raw, axis=1)
        scale = max(1.5 * np.median(error), 1e-5)
        new_weight = 1 / np.hypot(1, error / scale)
        if np.max(np.abs(new_weight - weight)) < 1e-5:
            break
        weight = new_weight
    clean = error < max(8 * np.median(error), 0.002)
    x, y = loads[clean], raw[clean]
    pilot = np.linalg.lstsq(a[clean], y, rcond=None)[0]
    terms = np.stack((x[:, _NEXT], x[:, _PREV], x * x[:, _NEXT], x**3), axis=-1)

    # Offsets and gains are channel-specific; normalized couplings are cyclic.
    def residual(p):
        return (p[:4] + p[4:8] * (x + terms @ p[8:]) - y).ravel()

    start = np.r_[pilot[0], np.diag(pilot[1:5]), np.zeros(4)]
    return least_squares(residual, start, x_scale='jac').x


def _forward(x, p):
    s, t, u, v = p[8:]
    xn = x[:, _NEXT]
    return p[:4] + p[4:8] * (x + s * xn + t * x[:, _PREV] + u * x * xn + v * x**3)


def _inverse(raw, p):
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return raw.copy()
    y = raw.reshape(-1, 4)
    s, t, u, v = p[8:]
    linear = np.eye(4)
    linear[_CHANNEL, _NEXT] = s
    linear[_CHANNEL, _PREV] = t
    x = np.linalg.solve(linear, ((y - p[:4]) / p[4:8]).T).T
    jac = np.broadcast_to(linear, (len(y), 4, 4)).copy()
    for _ in range(30):
        jac[:, _CHANNEL, _CHANNEL] = 1 + u * x[:, _NEXT] + 3 * v * x**2
        jac[:, _CHANNEL, _NEXT] = s + u * x
        step = np.linalg.solve(jac, (((_forward(x, p) - y) / p[4:8]))[..., None])[..., 0]
        x -= step / np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
        if np.max(np.abs(step)) < 1e-11:
            break
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a calibrated dataset without altering the input dictionaries or arrays."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    p = _fit_bridge(loads, raw)
    sweeps = calibration['single_channel_sweeps']
    output = dict(dataset)
    output['calibration_load_cells'] = dict(
        calibration, bridge_raw=np.asarray(calibration['true_channel_loads']).copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy()),
    )
    for key in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(key)
        if part is not None:
            output[key] = dict(part, past_channel_raw=_inverse(part['past_channel_raw'], p),
                               hold_channel_raw=_inverse(part['hold_channel_raw'], p))
    return output
