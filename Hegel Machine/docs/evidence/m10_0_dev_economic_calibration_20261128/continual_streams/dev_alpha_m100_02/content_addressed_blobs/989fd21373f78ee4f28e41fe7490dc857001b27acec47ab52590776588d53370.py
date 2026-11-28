"""Calibrate the four-channel load bridge in forward coordinates, then invert it."""

import numpy as np

_I, _J = np.triu_indices(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit_bridge(loads, raw):
    design = _basis(loads)
    coef = np.linalg.lstsq(design, raw, rcond=None)[0]
    # Row weights keep a simultaneous spike from biasing any channel.
    for _ in range(16):
        error = np.linalg.norm(raw - design @ coef, axis=1)
        scale = max(np.median(error), 1e-12)
        weight = 1.0 / np.hypot(1.0, error / (2.0 * scale))
        coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
    error = np.linalg.norm(raw - design @ coef, axis=1)
    good = error <= max(8.0 * np.median(error), 1e-10)
    a, y = design[good], raw[good]
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    # Retain measured nonlinear terms, not coefficients at the calibration noise floor.
    if len(a) > a.shape[1] and np.linalg.matrix_rank(a) == a.shape[1]:
        noise = np.linalg.norm(a @ coef - y, axis=0) / np.sqrt(len(a) - a.shape[1])
        leverage = np.linalg.norm(np.linalg.pinv(a), axis=1)
        active = np.abs(coef) > np.maximum(6.0 * leverage[:, None] * noise, 1e-12)
        active[:5] = True
        for j in range(4):
            keep = active[:, j]
            coef[~keep, j] = 0
            coef[keep, j] = np.linalg.lstsq(a[:, keep], y[:, j], rcond=None)[0]
    return coef


def _jacobian(x, coef):
    jac = np.broadcast_to(coef[1:5].T, (len(x), 4, 4)).copy()
    for k, (i, j) in enumerate(zip(_I, _J), start=5):
        jac[:, :, i] += x[:, j, None] * coef[k]
        jac[:, :, j] += x[:, i, None] * coef[k]
    jac += 3.0 * x[:, None, :]**2 * coef[15:].T
    return jac


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for _ in range(40):
        r = _basis(x) @ coef - target
        rows = np.flatnonzero(np.max(np.abs(r), axis=1) > 1e-11)
        if not rows.size:
            break
        base = x[rows]
        step = np.linalg.solve(_jacobian(base, coef), r[rows, :, None])[..., 0]
        error = np.linalg.norm(r[rows], axis=1)
        for i in range(18):
            trial = base - step * (0.5**i)
            improves = np.linalg.norm(_basis(trial) @ coef - target[rows], axis=1) < error
            x[rows[improves]] = trial[improves]
            if np.all(improves):
                break
            rows, base, step, error = (v[~improves] for v in (rows, base, step, error))
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in true-load coordinates without mutating the input."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    coef = _fit_bridge(loads, np.asarray(calibration['bridge_raw'], dtype=float))
    out = dict(dataset)
    out['calibration_load_cells'] = dict(calibration, bridge_raw=loads.copy())
    sweeps = calibration.get('single_channel_sweeps')
    if sweeps is not None:
        out['calibration_load_cells']['single_channel_sweeps'] = dict(
            sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy())
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            out[name] = dict(part, **{
                key: _inverse(part[key], coef)
                for key in ('past_channel_raw', 'hold_channel_raw')
            })
    return out
