"""Recover physical channel loads from the coupled load-cell bridge."""

import numpy as np


# After gain normalization, each bridge has the same response to its
# own load and to the previous and next loads in the four-channel ring.
_PREVIOUS = 0.15
_NEXT = -0.30
_PAIR = -0.22
_CUBIC = 0.18
_CHANNELS = np.arange(4)


def _bridge(x):
    prev = np.roll(x, 1, axis=-1)
    nxt = np.roll(x, -1, axis=-1)
    return x + _PREVIOUS * prev + _NEXT * nxt + _PAIR * x * nxt + _CUBIC * x**3


def _fit_bridge(loads, raw):
    response = _bridge(loads)

    def fit(weights):
        w = weights[:, None]
        mx = np.sum(w * response, axis=0) / weights.sum()
        my = np.sum(w * raw, axis=0) / weights.sum()
        dx = response - mx
        gain = np.sum(w * dx * (raw - my), axis=0) / np.sum(w * dx**2, axis=0)
        return my - mx * gain, gain

    offset, gain = fit(np.ones(len(loads)))
    for _ in range(30):
        err = np.linalg.norm(raw - (offset + gain * response), axis=1)
        scale = max(3e-4, 1.5 * np.median(err))
        offset, gain = fit(1 / (1 + (err / scale)**2))
    err = np.linalg.norm(raw - (offset + gain * response), axis=1)
    return fit(err < max(0.002, 8 * np.median(err)))


def _invert(raw, offset, gain):
    raw = np.asarray(raw, dtype=float)
    target = (raw.reshape(-1, 4) - offset) / gain
    x = target.copy()
    for _ in range(32):
        residual = _bridge(x) - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11
        if not active.any():
            return x.reshape(raw.shape)
        z = x[active]
        r = residual[active]
        jac = np.zeros((len(z), 4, 4))
        j = _CHANNELS
        jac[:, j, j] = 1 + _PAIR * np.roll(z, -1, axis=-1) + 3 * _CUBIC * z**2
        jac[:, j, (j - 1) % 4] = _PREVIOUS
        jac[:, j, (j + 1) % 4] = _NEXT + _PAIR * z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = z - step
        goal = target[active]
        norm = error[active]
        for _ in range(16):
            worse = np.linalg.norm(_bridge(trial) - goal, axis=1) >= norm
            if not worse.any():
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    raise RuntimeError('Load-cell bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy with train, test and verification loads in physical units."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    offset, gain = _fit_bridge(loads, np.asarray(cal['bridge_raw'], dtype=float))
    result = dataset.copy()
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells'] = {
        **cal,
        'bridge_raw': cal['true_channel_loads'].copy(),
        'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
    }
    for key in ('train', 'test', 'chamber_verification_trials'):
        group = dataset.get(key)
        if group is not None:
            result[key] = {
                **group,
                'past_channel_raw': _invert(group['past_channel_raw'], offset, gain),
                'hold_channel_raw': _invert(group['hold_channel_raw'], offset, gain),
            }
    return result
