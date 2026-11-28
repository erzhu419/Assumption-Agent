"""Calibrate the coupled load-cell bridge before constitutive fitting."""
import numpy as np


_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4
_LOAD_FIELDS = ('past_channel_raw', 'hold_channel_raw')


def _design(loads, channel):
    own = loads[:, channel]
    return np.column_stack((np.ones(len(loads)), loads,
                            own * loads[:, _NEXT[channel]], own**3))


def _fit_channel(a, y):
    """Downweight gross readout spikes, then refit to the clean readings."""
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(30):
        residual = a @ coef - y
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(2.5 * 1.4826 * mad, 1e-9)
        weights = 1 / np.hypot(1, residual / scale)
        coef = np.linalg.lstsq(a * weights[:, None], y * weights, rcond=None)[0]
    residual = a @ coef - y
    centered = residual - np.median(residual)
    keep = np.abs(centered) <= max(8 * 1.4826 * np.median(np.abs(centered)), 1e-8)
    return np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]


def _fit_bridge(loads, bridge, sweeps):
    coefficients = []
    labs = np.asarray(sweeps['lab_id'])
    for j in _CHANNELS:
        use = labs == j
        axes = np.zeros((np.count_nonzero(use), 4))
        axes[:, j] = np.asarray(sweeps['true_stress'])[use]
        a = _design(np.concatenate((loads, axes)), j)
        y = np.concatenate((bridge[:, j], np.asarray(sweeps['cmd_stress'])[use]))
        coefficients.append(_fit_channel(a, y))
    return np.column_stack(coefficients)


def _forward(x, coef):
    return coef[0] + x @ coef[1:5] + coef[5] * x * x[:, _NEXT] + coef[6] * x**3


def _inverse(raw, coef):
    shape = np.shape(raw)
    target = np.asarray(raw, dtype=float).reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for _ in range(35):
        residual = _forward(x, coef) - target
        error = np.max(np.abs(residual), axis=1)
        active = error > 1e-11
        if not np.any(active):
            break
        z = x[active]
        jac = np.broadcast_to(coef[1:5].T, (len(z), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += coef[5] * z[:, _NEXT] + 3 * coef[6] * z**2
        jac[:, _CHANNELS, _NEXT] += coef[5] * z
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        goals = target[active]
        size = error[active]
        trial = z - step
        for _ in range(16):
            worse = np.max(np.abs(_forward(trial, coef) - goals), axis=1) >= size
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    return x.reshape(shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return canonical loads without modifying the supplied dataset."""
    cal = dataset['calibration_load_cells']
    sweeps = cal['single_channel_sweeps']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    bridge = np.asarray(cal['bridge_raw'], dtype=float)
    coef = _fit_bridge(loads, bridge, sweeps)
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=np.asarray(cal['true_channel_loads']).copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy()),
    )
    for key in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(key)
        if part is not None:
            result[key] = dict(part, **{field: _inverse(part[field], coef) for field in _LOAD_FIELDS})
    return result
