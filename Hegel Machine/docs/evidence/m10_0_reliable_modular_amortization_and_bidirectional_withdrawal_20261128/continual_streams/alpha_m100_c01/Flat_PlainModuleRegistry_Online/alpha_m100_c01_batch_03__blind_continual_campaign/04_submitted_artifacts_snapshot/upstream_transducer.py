"""Calibrate coupled load bridges before fitting the chamber response."""

import numpy as np


_I, _J = np.triu_indices(4)
_EYE = np.eye(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit_forward(loads, raw):
    design = _basis(loads)
    coef = np.linalg.lstsq(design, raw, rcond=None)[0]
    # Downweight complete rows: a calibration spike can affect all four bridges.
    scale = np.max(np.quantile(raw, .9, axis=0) - np.quantile(raw, .1, axis=0)) / 4
    for frac in (.4, .1, .02, .004, .001, .00035, .0002):
        cutoff = max(frac * scale, np.finfo(float).eps)
        for _ in range(15):
            error = np.linalg.norm(raw - design @ coef, axis=1)
            weight = cutoff / np.hypot(cutoff, error)
            coef = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
    error = np.linalg.norm(raw - design @ coef, axis=1)
    # Once isolated, spikes carry no information about the clean forward map.
    clean = error <= 8 * max(np.median(error), np.finfo(float).eps * max(1., scale))
    clean_coef, _, rank, _ = np.linalg.lstsq(design[clean], raw[clean], rcond=None)
    return clean_coef if rank == design.shape[1] else coef


def _inverse(raw, coef):
    shape = raw.shape
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    if not len(y):
        return y.reshape(shape)
    linear = coef[1:5]
    quadratic = np.zeros((4, 4, 4))
    quadratic[_I, _J] += coef[5:15]
    quadratic[_J, _I] += coef[5:15]
    cubic = coef[15:]
    x = np.linalg.solve(linear.T, (y - coef[0]).T).T
    tolerance = 1e-11 * (1 + np.linalg.norm(y, axis=1))
    for _ in range(40):
        resid = _basis(x) @ coef - y
        error = np.linalg.norm(resid, axis=1)
        active = np.flatnonzero(error > tolerance)
        if not len(active):
            break
        z = x[active]
        deriv = (linear + np.einsum('nk,kio->nio', z, quadratic)
                 + (3 * z**2)[:, :, None] * _EYE * cubic)
        step = np.linalg.solve(deriv.transpose(0, 2, 1), resid[active, :, None])[..., 0]
        for _ in range(24):
            trial = x[active] - step
            better = np.linalg.norm(_basis(trial) @ coef - y[active], axis=1) < error[active]
            x[active[better]] = trial[better]
            active = active[~better]
            if not len(active):
                break
            step = .5 * step[~better]
    if np.any(np.linalg.norm(_basis(x) @ coef - y, axis=1) > 10 * tolerance):
        raise ValueError('Load bridge inverse did not converge')
    return x.reshape(shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy with all load histories and sweeps in canonical coordinates."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    coef = _fit_forward(loads, np.asarray(calibration['bridge_raw'], dtype=float))

    result = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(key)
        if split is not None:
            result[key] = dict(split, **{
                field: _inverse(np.asarray(split[field]), coef)
                for field in ('past_channel_raw', 'hold_channel_raw')
            })

    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=calibration['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    return result
