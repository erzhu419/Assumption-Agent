"""Calibrate the four-channel load bridge before fitting material response."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import root

# Column 4 is unity, so every monomial can use three factor slots.
_TERMS = np.array([term + (4,) * (3 - degree)
                   for degree in range(4)
                   for term in combinations_with_replacement(range(4), degree)])


def _basis(loads):
    padded = np.concatenate((loads, np.ones(loads.shape[:-1] + (1,))), axis=-1)
    return padded[..., _TERMS].prod(axis=-1)


def _fit_bridge(loads, raw):
    design = _basis(loads)
    affine = design[:, :5]
    coef = np.linalg.lstsq(affine, raw, rcond=None)[0]
    # Forward fitting puts the spikes in the response, not the predictors.
    for _ in range(12):
        err = np.linalg.norm(raw - affine @ coef, axis=1)
        scale = max(2 * np.median(err), 1e-10)
        w = 1 / np.hypot(1, err / scale)
        coef = np.linalg.lstsq(affine * w[:, None], raw * w[:, None], rcond=None)[0]
    err = np.linalg.norm(raw - affine @ coef, axis=1)
    keep = err <= max(8 * np.median(err), 1e-9)
    coef = np.linalg.lstsq(design[keep], raw[keep], rcond=None)[0]

    # Reclassify against the nonlinear model so genuine curvature does not
    # get discarded as a spike by the affine pilot.
    err = np.linalg.norm(raw - design @ coef, axis=1)
    keep = err <= max(8 * np.median(err), 1e-9)
    a, y = design[keep], raw[keep]
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    residual = y - a @ coef
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    se = np.linalg.norm(np.linalg.pinv(a), axis=1)[:, None] * noise
    # Drop nonlinear terms indistinguishable from calibration noise, then
    # refit each output without shrinking its supported coefficients.
    for j in range(4):
        use = np.abs(coef[:, j]) > np.maximum(6 * se[:, j], 1e-10)
        use[:5] = True
        coef[:, j] = 0
        coef[use, j] = np.linalg.lstsq(a[:, use], y[:, j], rcond=None)[0]
    return coef


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    rows = raw.reshape(-1, 4)
    starts = np.linalg.solve(coef[1:5].T, (rows - coef[0]).T).T
    result = np.empty_like(rows)
    for i, (target, start) in enumerate(zip(rows, starts)):
        def residual(loads):
            return _basis(loads) @ coef - target

        sol = root(residual, start, options={'xtol': 1e-10})
        if not np.max(np.abs(sol.fun)) <= 1e-8 * (1 + np.max(np.abs(target))):
            raise ValueError('Load bridge inversion did not converge')
        result[i] = sol.x
    return result.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a dataset with all bridge readings in true load coordinates."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    coef = _fit_bridge(loads, np.asarray(cal['bridge_raw'], dtype=float))
    result = dataset.copy()
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            result[name] = dict(part,
                                past_channel_raw=_invert(part['past_channel_raw'], coef),
                                hold_channel_raw=_invert(part['hold_channel_raw'], coef))
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    return result
