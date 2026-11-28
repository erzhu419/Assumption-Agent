"""Fit the coupled forward bridge, then recover loads in canonical coordinates."""
from itertools import combinations_with_replacement

import numpy as np


# Input exponents for all monomials through degree three, including the offset.
_POWERS = np.array([
    np.bincount(term, minlength=4)
    for degree in range(4)
    for term in combinations_with_replacement(range(4), degree)
])
_DERIV_POWERS = np.maximum(_POWERS[None, :, :] - np.eye(4, dtype=int)[:, None, :], 0)
_DERIV_FACTOR = _POWERS.T


def _basis(x):
    return np.prod(x[:, None, :] ** _POWERS, axis=-1)


def _fit(loads, bridge):
    a = _basis(loads)
    if np.array_equal(loads, bridge):
        coef = np.zeros((a.shape[1], 4))
        coef[1:5] = np.eye(4)
        return coef

    # A spike is a bad observation of the whole bridge, not four independent
    # errors. Give the row one Cauchy weight using its multichannel residual.
    w = np.ones((len(a), 1))
    for _ in range(40):
        coef = np.linalg.lstsq(w * a, w * bridge, rcond=None)[0]
        residual = a @ coef - bridge
        radius = np.linalg.norm(residual, axis=1)
        scale = max(np.median(radius), 1e-12)
        w = (1 / np.hypot(1, radius / (2 * scale)))[:, None]

    good = radius < 8 * scale
    if np.linalg.matrix_rank(a[good]) < a.shape[1]:
        return coef
    # Compact each output separately; compare coefficients on their own
    # uncertainty scales so differently scaled monomials are treated alike.
    clean_a = a[good]
    sensitivity = np.linalg.norm(np.linalg.pinv(clean_a), axis=1)
    full = np.linalg.lstsq(clean_a, bridge[good], rcond=None)[0]
    noise = 1.4826 * np.median(abs(clean_a @ full - bridge[good]), axis=0)
    keep = abs(full) > 7 * sensitivity[:, None] * np.maximum(noise, 1e-12)
    keep[:5] = True
    compact = np.zeros_like(coef)
    for j, support in enumerate(keep.T):
        compact[support, j] = np.linalg.lstsq(clean_a[:, support], bridge[good, j], rcond=None)[0]
    return compact


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T
    tolerance = 1e-11 * (1 + np.linalg.norm(y, axis=1))
    for _ in range(40):
        r = _basis(x) @ coef - y
        error = np.linalg.norm(r, axis=1)
        active = np.flatnonzero(error > tolerance)
        if not len(active):
            return x.reshape(raw.shape)
        z = x[active]
        d = np.prod(z[:, None, None, :] ** _DERIV_POWERS, axis=-1) * _DERIV_FACTOR
        jacobian = (d @ coef).swapaxes(1, 2)
        step = np.linalg.solve(jacobian, r[active, :, None])[..., 0]
        # Backtracking keeps Newton on the locally improving branch when a
        # reading is well outside the bridge's linear range.
        for _ in range(16):
            trial = z - step
            improved = np.linalg.norm(_basis(trial) @ coef - y[active], axis=1) < error[active]
            x[active[improved]] = trial[improved]
            if improved.all():
                break
            active, z, step = active[~improved], z[~improved], step[~improved] / 2
    if np.any(np.linalg.norm(_basis(x) @ coef - y, axis=1) > tolerance):
        raise ValueError('Coupled bridge inversion did not converge')
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a calibrated dataset without modifying its source arrays."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    coef = _fit(loads, np.asarray(cal['bridge_raw'], dtype=float))
    out = dict(dataset)
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(cal['single_channel_sweeps'],
                                  cmd_stress=cal['single_channel_sweeps']['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(name)
        if block is not None:
            out[name] = dict(block, **{
                field: _invert(block[field], coef)
                for field in ('past_channel_raw', 'hold_channel_raw')
            })
    return out
