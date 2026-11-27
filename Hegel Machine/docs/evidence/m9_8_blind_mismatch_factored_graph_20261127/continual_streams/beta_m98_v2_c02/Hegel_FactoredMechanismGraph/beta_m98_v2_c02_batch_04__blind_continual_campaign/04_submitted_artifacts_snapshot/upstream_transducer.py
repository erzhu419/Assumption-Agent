"""Robust rosette calibration in the downstream solver's reference coordinates."""

import numpy as np

_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4


def _forward(ref, coef):
    return (coef[:, 0] + ref @ coef[:, 1:5].T
            + coef[:, 5] * ref * ref[:, _NEXT]
            + coef[:, 6] * ref**3)


def _fit(ref, raw):
    # Use shared row weights: an EMI event contaminates the rosette jointly.
    design = [np.column_stack((np.ones(len(ref)), ref,
                               ref[:, j] * ref[:, k], ref[:, j]**3))
              for j, k in zip(_CHANNEL, _NEXT)]

    def weighted_fit(weight):
        return np.array([np.linalg.lstsq(a * weight[:, None], raw[:, j] * weight,
                                         rcond=None)[0]
                         for j, a in enumerate(design)])

    weight = np.ones(len(ref))
    for _ in range(25):
        coef = weighted_fit(weight)
        residual = np.linalg.norm(_forward(ref, coef) - raw, axis=1)
        median = np.median(residual)
        weight = 1 / np.hypot(1, residual / max(1e-7, 2 * median))
    # Refit the measured response without the spike rows or Cauchy downweighting.
    return weighted_fit((residual <= max(1e-6, 8 * median)).astype(float))


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    linear = coef[:, 1:5]
    ref = np.linalg.solve(linear, (y - coef[:, 0]).T).T
    for _ in range(32):
        residual = _forward(ref, coef) - y
        error = np.linalg.norm(residual, axis=1)
        if np.all(error <= 1e-11 * (1 + np.linalg.norm(y, axis=1))):
            break
        jac = np.broadcast_to(linear, (len(y), 4, 4)).copy()
        jac[:, _CHANNEL, _CHANNEL] += (
            coef[:, 5] * ref[:, _NEXT] + 3 * coef[:, 6] * ref**2)
        jac[:, _CHANNEL, _NEXT] += coef[:, 5] * ref
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        # Backtrack only rows where a full Newton step increases the residual.
        trial = ref - step
        for _ in range(12):
            bad = np.linalg.norm(_forward(trial, coef) - y, axis=1) > error
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = ref[bad] - step[bad]
        ref = trial
    return ref.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Calibrate observations and pulses without modifying the input study."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    banks = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(banks, raw_readout=np.asarray(banks['ref_signal']).copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            for pulse in pulses]
    return calibrated
