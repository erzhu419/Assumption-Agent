"""Calibrate the four-channel bridge before fitting the material law."""

import numpy as np

_I, _J = np.triu_indices(4)
_BLOCKS = ('train', 'test', 'chamber_verification_trials')
_LOADS = ('past_channel_raw', 'hold_channel_raw')


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit_bridge(loads, raw):
    a = _basis(loads)
    weights = np.ones(len(a))
    # A spike affects several readings at once, so weight the whole row.
    for _ in range(20):
        w = np.sqrt(weights[:, None])
        coef = np.linalg.lstsq(a * w, raw * w, rcond=None)[0]
        err = np.linalg.norm(raw - a @ coef, axis=1)
        scale = max(np.median(err), 1e-12)
        weights = 1 / (1 + (err / (2.5 * scale))**2)
    # Refit at full weight once the gross spikes are isolated.
    keep = err <= 6 * scale
    if np.linalg.matrix_rank(a[keep]) == a.shape[1]:
        coef = np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]
    return coef


def _invert_bridge(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    linear = coef[1:5].T
    x = np.linalg.solve(linear, (target - coef[0]).T).T.copy()
    # Jacobian axes are (row, output channel, input channel).
    quadratic = np.zeros((4, 4, 4))
    quadratic[:, _I, _J] += coef[5:15].T
    quadratic[:, _J, _I] += coef[5:15].T
    for _ in range(24):
        r = _basis(x) @ coef - target
        error = np.max(np.abs(r), axis=1, keepdims=True)
        if np.all(error <= 1e-11):
            break
        jac = linear + np.einsum('oij,nj->noi', quadratic, x)
        jac += coef[15:].T * (3 * x[:, None, :]**2)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        # Shorten a Newton step only on rows that do not improve.
        trial = x - step
        for _ in range(12):
            bad = np.max(np.abs(_basis(trial) @ coef - target), axis=1) > error[:, 0]
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = x[bad] - step[bad]
        x = trial
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a canonical-coordinate dataset without altering the input."""
    cal = dataset['calibration_load_cells']
    true = cal['true_channel_loads']
    coef = _fit_bridge(true, cal['bridge_raw'])
    sweeps = cal['single_channel_sweeps']
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=true.copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for key in _BLOCKS:
        block = dataset.get(key)
        if block is not None:
            result[key] = dict(block, **{f: _invert_bridge(block[f], coef) for f in _LOADS})
    return result
