"""Calibrate the four-channel load bridge in the known-load direction."""
import numpy as np

_I, _J = np.triu_indices(4)
_EYE = np.eye(4)


def _design(x):
    return np.column_stack((np.ones(len(x)), x, x[:, _I] * x[:, _J], x**3))


def _fit_forward(x, raw):
    design = _design(x)
    coeff = np.linalg.lstsq(design, raw, rcond=None)[0]
    # Give each calibration row one weight: a spike on any bridge output
    # invalidates the simultaneous reading. Anneal to separate noise from spikes.
    for scale in (0.8, 0.2, 0.06, 0.02, 0.006, 0.002, 0.0006):
        for _ in range(16):
            residual = design @ coeff - raw
            weight = 1 / np.sqrt(1 + np.sum((residual / scale)**2, axis=1))
            update = np.linalg.lstsq(design * weight[:, None], raw * weight[:, None], rcond=None)[0]
            if np.max(np.abs(update - coeff)) < 1e-10:
                coeff = update
                break
            coeff = update
    return coeff


def _jacobian(x, coeff):
    quad = x[:, _I, None] * _EYE[_J] + x[:, _J, None] * _EYE[_I]
    return (coeff[1:5] + np.einsum('nki,kj->nij', quad, coeff[5:15])
            + 3 * x[:, :, None]**2 * coeff[15:19])


def _invert(raw, coeff):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    x = np.linalg.solve(coeff[1:5].T, (target - coeff[0]).T).T
    for _ in range(35):
        residual = _design(x) @ coeff - target
        error = np.sum(residual**2, axis=1)
        active = error > 1e-22
        if not np.any(active):
            break
        xa, ta, ea = x[active], target[active], error[active]
        jac = _jacobian(xa, coeff).transpose(0, 2, 1)
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = xa - step
        for _ in range(14):
            worse = np.sum((_design(trial) @ coeff - ta)**2, axis=1) >= ea
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = xa[worse] - step[worse]
        improved = np.sum((_design(trial) @ coeff - ta)**2, axis=1) < ea
        if not np.any(improved):
            break
        xa[improved] = trial[improved]
        x[active] = xa
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy with bridge readings and calibration commands in true units."""
    cal = dataset['calibration_load_cells']
    true = cal['true_channel_loads']
    coeff = _fit_forward(np.asarray(true, dtype=float), np.asarray(cal['bridge_raw'], dtype=float))
    corrected = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(key)
        if block is not None:
            corrected[key] = dict(block,
                                  past_channel_raw=_invert(block['past_channel_raw'], coeff),
                                  hold_channel_raw=_invert(block['hold_channel_raw'], coeff))
    sweeps = cal['single_channel_sweeps']
    corrected['calibration_load_cells'] = dict(
        cal, bridge_raw=true.copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    return corrected
