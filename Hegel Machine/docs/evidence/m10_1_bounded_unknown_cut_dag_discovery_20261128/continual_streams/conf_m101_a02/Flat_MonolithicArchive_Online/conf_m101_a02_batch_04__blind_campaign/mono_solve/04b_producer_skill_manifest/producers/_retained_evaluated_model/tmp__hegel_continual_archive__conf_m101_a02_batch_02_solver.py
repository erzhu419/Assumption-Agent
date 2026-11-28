"""Calibrate the inputs, then fit the scalar and time-course readouts."""

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration(panel):
    """Estimate the four gains and offsets after trimming gross row outliers."""
    u, v = panel['u'], panel['v']
    design = _affine(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        coef = _fit(design[keep], v[keep])
        error = np.linalg.norm(v - design @ coef, axis=1)
        keep = error <= 4 * np.median(error)
    return np.array([_fit(_affine(uj[keep]), vj[keep])
                     for uj, vj in zip(u.T, v.T)]).T


def _label_design(z, frequency):
    z0, z1, z2, z3 = z.T
    contrast = z0 + z1 - z2 - z3
    pair = z0 * z2 + z1 * z3
    return np.column_stack((_affine(z), pair, np.sin(frequency * contrast),
                            np.tanh(contrast), pair / (1 + contrast**2)))


def solve_batch(study: dict) -> dict:
    offset, gain = _calibration(study['panel_a'])
    panel = study['panel_b']
    mixer = _fit(_affine(panel['u']), panel['v'])
    train = study['train']
    z = _affine((train['x'] - offset) / gain) @ mixer
    z_test = _affine((study['test']['x'] - offset) / gain) @ mixer
    frequency = study['telemetry_vec'][0]
    label_coef = _fit(_label_design(z, frequency), train['y'])
    y_hat = _label_design(z_test, frequency) @ label_coef
    state_pred = _affine(z_test) @ _fit(_affine(z), train['traj'])
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
