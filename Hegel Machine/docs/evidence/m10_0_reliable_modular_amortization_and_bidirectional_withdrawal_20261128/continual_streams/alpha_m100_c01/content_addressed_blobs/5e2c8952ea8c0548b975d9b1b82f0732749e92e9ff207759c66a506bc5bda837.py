"""Calibrate the four-channel bridge before the constitutive fit."""
from itertools import combinations

import numpy as np

_PAIRS = tuple(combinations(range(4), 2))


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x**2, x**3,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS)))


def _fit_bridge(x, raw):
    a = _basis(x)
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    for _ in range(50):
        row_error = np.linalg.norm(a @ coef - raw, axis=1)
        scale = max(np.median(row_error), 1e-10)
        weight = 1 / np.hypot(1, row_error / (2 * scale))
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
    row_error = np.linalg.norm(a @ coef - raw, axis=1)
    good = row_error <= max(6 * np.median(row_error), 1e-9)
    a, raw = a[good], raw[good]
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    # Select terms above calibration noise separately for each output. Keep
    # the affine block so the inversion has a stable starting approximation.
    error = a @ coef - raw
    noise = np.linalg.norm(error, axis=0) / np.sqrt(max(len(a) - a.shape[1], 1))
    uncertainty = np.linalg.norm(np.linalg.pinv(a), axis=1)[:, None] * noise
    for channel in range(4):
        use = np.abs(coef[:, channel]) > 6 * uncertainty[:, channel]
        use[:5] = True
        coef[:, channel] = 0
        coef[use, channel] = np.linalg.lstsq(a[:, use], raw[:, channel], rcond=None)[0]
    return coef


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    linear = coef[1:5]
    x = np.linalg.solve(linear.T, (y - coef[0]).T).T
    for _ in range(30):
        residual = _basis(x) @ coef - y
        distance = np.linalg.norm(residual, axis=1)
        active = distance > 1e-11 * (1 + np.linalg.norm(y, axis=1))
        if not active.any():
            break
        z = x[active]
        # Jacobian with input index first, like the affine coefficient block.
        jac = linear + 2 * z[..., None] * coef[5:9] + 3 * z[..., None]**2 * coef[9:13]
        for (i, j), term in zip(_PAIRS, coef[13:]):
            jac[:, i] += z[:, j, None] * term
            jac[:, j] += z[:, i, None] * term
        step = np.linalg.solve(jac.transpose(0, 2, 1), residual[active, :, None])[..., 0]
        target = y[active]
        trial = z - step
        for _ in range(16):
            bad = np.linalg.norm(_basis(trial) @ coef - target, axis=1) > distance[active]
            if not bad.any():
                break
            step[bad] *= 0.5
            trial[bad] = z[bad] - step[bad]
        else:
            bad = np.linalg.norm(_basis(trial) @ coef - target, axis=1) > distance[active]
            trial[bad] = z[bad]
        x[active] = trial
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    out = dict(dataset)
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    identity = np.array_equal(raw, loads)
    coef = None if identity else _fit_bridge(loads, raw)

    def convert(split):
        return {**split, **{key: (np.asarray(split[key]).copy() if identity else _inverse(split[key], coef))
                            for key in ('past_channel_raw', 'hold_channel_raw')}}

    for key in ('train', 'test'):
        out[key] = convert(dataset[key])
    trials = dataset.get('chamber_verification_trials')
    if trials is not None:
        out['chamber_verification_trials'] = (convert(trials) if isinstance(trials, dict)
                                               else [convert(trial) for trial in trials])
    sweeps = cal['single_channel_sweeps']
    out['calibration_load_cells'] = {
        **cal, 'bridge_raw': loads.copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': np.asarray(sweeps['true_stress']).copy()},
    }
    return out
