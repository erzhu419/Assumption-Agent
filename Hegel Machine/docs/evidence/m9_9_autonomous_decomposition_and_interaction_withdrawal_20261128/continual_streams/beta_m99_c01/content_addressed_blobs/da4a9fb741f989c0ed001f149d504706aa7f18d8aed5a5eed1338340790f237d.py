"""Fit the rosette in forward coordinates and undo it before PDE inversion."""

import numpy as np
from scipy.optimize import least_squares


_NEXT = np.array([1, 2, 3, 0])


def _fit_forward(ref, raw):
    """Fit linear coupling, channel cubics, and cyclic pair interactions."""
    n = len(ref)
    common = np.column_stack((np.ones(n), ref))
    design = np.stack([
        np.column_stack((common, ref[:, j]**3, ref[:, j] * ref[:, k]))
        for j, k in enumerate(_NEXT)
    ])

    def fit(weight):
        return np.stack([
            np.linalg.lstsq(a * weight[:, None], y * weight, rcond=None)[0]
            for a, y in zip(design, raw.T)
        ])

    # EMI spikes affect entire rows. Downweight them jointly before the final
    # clean-row refit so the large raw readings cannot set the calibration.
    coef = fit(np.ones(n))
    magnitude = max(1., np.median(np.abs(raw)))
    for _ in range(20):
        residual = np.linalg.norm(np.einsum('jnp,jp->nj', design, coef) - raw, axis=1)
        scale = max(np.median(residual), 1e-12 * magnitude)
        weight = 1. / np.hypot(1., residual / (2.5 * scale))
        updated = fit(weight)
        if np.max(np.abs(updated - coef)) <= 1e-11 * max(1., np.max(np.abs(coef))):
            coef = updated
            break
        coef = updated
    residual = np.linalg.norm(np.einsum('jnp,jp->nj', design, coef) - raw, axis=1)
    keep = residual <= max(6. * np.median(residual), 1e-11 * magnitude)
    return fit(keep.astype(float)) if keep.sum() >= design.shape[-1] else coef


def _inverse(values, coef):
    raw = np.asarray(values, dtype=float)
    rows = raw.reshape(-1, 4)
    offset, linear, cube, pair = coef[:, 0], coef[:, 1:5], coef[:, 5], coef[:, 6]
    seeds = np.linalg.solve(linear, (rows - offset).T).T
    j = np.arange(4)

    def jac(x):
        a = linear.copy()
        a[j, j] += 3. * cube * x**2 + pair * x[_NEXT]
        a[j, _NEXT] += pair * x
        return a

    def forward(x):
        return offset + linear @ x + cube * x**3 + pair * x * x[_NEXT]

    out = np.empty_like(rows)
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        out[i] = least_squares(lambda x: forward(x) - row, seed, jac=jac,
                               gtol=1e-12, xtol=1e-12, ftol=1e-12).x
    return out.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    coef = _fit_forward(ref, raw)
    single = sweeps['single_bank_sweeps']
    result = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(single, raw_readout=single['ref_signal'].copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            for pulse in pulses
        ]
    return result
