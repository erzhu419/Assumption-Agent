"""Recover canonical load coordinates from the four-channel bridge."""

import numpy as np


# Include all quadratic interactions in the discovery fit. The measured
# cubic response acts on individual channels, not mixed cubic monomials.
_TERMS = ((), (0,), (1,), (2,), (3,)) + tuple(
    (i, j) for i in range(4) for j in range(i, 4)
) + tuple((i, i, i) for i in range(4))


def _basis(loads):
    return np.column_stack([np.prod(loads[:, term], axis=1) for term in _TERMS])


def _fit_bridge(loads, raw):
    design = _basis(loads)
    weights = np.ones(len(design))
    for _ in range(24):
        root = np.sqrt(weights[:, None])
        coef = np.linalg.lstsq(root * design, root * raw, rcond=None)[0]
        error = raw - design @ coef
        scale = np.maximum(1.4826 * np.median(
            np.abs(error - np.median(error, axis=0)), axis=0), 1e-10)
        weights = 1 / (1 + np.mean((error / (2.5 * scale)) ** 2, axis=1))

    # Remove unsupported coefficients and refit on the common inlier rows.
    # Measured spike rows have residuals orders of magnitude above bridge noise.
    inliers = np.max(np.abs(error) / scale, axis=1) < 7
    clean = design[inliers]
    rms = np.sqrt(np.mean(clean ** 2, axis=0))
    for channel in range(4):
        keep = np.abs(coef[:, channel]) * rms > 8 * scale[channel]
        keep[0] = True
        reduced = clean[:, keep]
        if np.linalg.matrix_rank(reduced) == keep.sum():
            sparse = np.linalg.lstsq(reduced, raw[inliers, channel], rcond=None)[0]
            if np.sqrt(np.mean((reduced @ sparse - raw[inliers, channel]) ** 2)) <= 1.5 * scale[channel]:
                coef[:, channel] = 0
                coef[keep, channel] = sparse
    return coef


def _jacobian(loads, coef):
    jac = np.broadcast_to(coef[1:5].T, (len(loads), 4, 4)).copy()
    for term, coeff in zip(_TERMS[5:], coef[5:]):
        if not np.any(coeff):
            continue
        for k, j in enumerate(term):
            other = term[:k] + term[k + 1:]
            jac[:, :, j] += np.prod(loads[:, other], axis=1)[:, None] * coeff
    return jac


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    loads = np.linalg.solve(coef[1:5].T, (target - coef[0]).T).T
    for _ in range(35):
        error = _basis(loads) @ coef - target
        norm = np.max(np.abs(error), axis=1)
        active = norm > 1e-11
        if not np.any(active):
            break
        start = loads[active]
        residual = error[active]
        rhs = target[active]
        baseline = norm[active]
        step = np.linalg.solve(_jacobian(start, coef), residual[..., None])[..., 0]
        trial = start - step
        for _ in range(16):
            worse = np.max(np.abs(_basis(trial) @ coef - rhs), axis=1) >= baseline
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = start[worse] - step[worse]
        loads[active] = trial
    return loads.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a calibrated copy, leaving the input and other fields intact."""
    calibration = dataset['calibration_load_cells']
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    coef = _fit_bridge(loads, np.asarray(calibration['bridge_raw'], dtype=float))
    result = dict(dataset)
    for name in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(name)
        if block is not None:
            result[name] = dict(block)
            for field in ('past_channel_raw', 'hold_channel_raw'):
                result[name][field] = _invert(block[field], coef)
    sweeps = calibration['single_channel_sweeps']
    result['calibration_load_cells'] = dict(
        calibration, bridge_raw=calibration['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    return result
