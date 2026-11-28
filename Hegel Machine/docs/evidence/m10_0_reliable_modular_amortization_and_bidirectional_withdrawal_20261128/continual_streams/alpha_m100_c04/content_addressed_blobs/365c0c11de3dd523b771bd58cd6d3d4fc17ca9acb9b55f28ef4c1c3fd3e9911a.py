"""Fit the four load-cell bridge channels and express trial loads in physical units."""

import numpy as np


_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4
_PREV = (_CHANNEL - 1) % 4


def _fit_bridge(load, raw):
    # Each channel has a cubic self-response, linear neighbor coupling,
    # and one bilinear interaction with its next neighbor.
    design = np.stack((np.ones_like(load), load, load[:, _NEXT],
                       load[:, _PREV], load**3, load * load[:, _NEXT]), axis=1)
    weight = np.ones(len(load))
    for _ in range(40):
        coef = np.column_stack([
            np.linalg.lstsq(design[:, :, j] * weight[:, None],
                            raw[:, j] * weight, rcond=None)[0]
            for j in _CHANNEL
        ])
        residual = np.linalg.norm(np.sum(design * coef, axis=1) - raw, axis=1)
        # A shared row weight keeps a spike in any channel from biasing the fit.
        scale = max(2.5 * np.median(residual), 1e-9)
        new_weight = 1.0 / np.hypot(1.0, residual / scale)
        if np.max(np.abs(weight - new_weight)) < 1e-7:
            break
        weight = new_weight
    return coef


def _forward(load, coef):
    offset, gain, nxt, prev, cubic, cross = coef
    neighbor = load[:, _NEXT]
    return (offset + gain * load + nxt * neighbor + prev * load[:, _PREV]
            + cubic * load**3 + cross * load * neighbor)


def _inverse(raw, coef):
    shape = np.shape(raw)
    raw = np.asarray(raw, dtype=float).reshape(-1, 4)
    if not len(raw):
        return raw.reshape(shape)
    offset, gain, nxt, prev, cubic, cross = coef
    linear = np.zeros((4, 4))
    linear[_CHANNEL, _CHANNEL] = gain
    linear[_CHANNEL, _NEXT] = nxt
    linear[_CHANNEL, _PREV] = prev
    load = np.linalg.solve(linear, (raw - offset).T).T
    for _ in range(30):
        residual = _forward(load, coef) - raw
        error = np.linalg.norm(residual, axis=1)
        if error.max() < 1e-11:
            break
        jac = np.broadcast_to(linear, (len(raw), 4, 4)).copy()
        jac[:, _CHANNEL, _CHANNEL] += 3 * cubic * load**2 + cross * load[:, _NEXT]
        jac[:, _CHANNEL, _NEXT] += cross * load
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        trial = load - step
        # Damp only the rows for which the full Newton step increases error.
        for _ in range(12):
            worse = np.linalg.norm(_forward(trial, coef) - raw, axis=1) > error
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = load[worse] - step[worse]
        load = trial
    return load.reshape(shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy with trial loads and calibration references in identity coordinates."""
    calibration = dataset['calibration_load_cells']
    true_loads = calibration['true_channel_loads']
    coef = _fit_bridge(np.asarray(true_loads, dtype=float),
                       np.asarray(calibration['bridge_raw'], dtype=float))
    sweeps = calibration['single_channel_sweeps']
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=true_loads.copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            result[name] = dict(part, past_channel_raw=_inverse(part['past_channel_raw'], coef),
                                hold_channel_raw=_inverse(part['hold_channel_raw'], coef))
    return result
