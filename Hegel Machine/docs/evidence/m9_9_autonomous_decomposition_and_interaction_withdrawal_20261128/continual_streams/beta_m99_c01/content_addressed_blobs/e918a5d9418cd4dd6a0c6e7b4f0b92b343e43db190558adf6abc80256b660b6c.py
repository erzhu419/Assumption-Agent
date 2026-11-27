"""Robust calibration of the four-channel rosette before PDE inversion."""
import numpy as np


def _fit_rosette(ref, raw):
    # Each output has affine cross-talk, a self-cubic, and a cyclic pair term.
    base = np.column_stack((np.ones(len(ref)), ref))
    adjacent = np.roll(ref, -1, axis=1)
    design = [np.column_stack((base, ref[:, j] ** 3, ref[:, j] * adjacent[:, j]))
              for j in range(4)]
    # Initialize away from gross EMI rows, then weight by joint residual.
    amplitude = np.max(np.abs(raw), axis=1)
    weight = (amplitude <= 3 * np.quantile(amplitude, .9)).astype(float)
    for _ in range(12):
        sw = np.sqrt(weight)
        coef = np.stack([np.linalg.lstsq(d * sw[:, None], raw[:, j] * sw,
                                        rcond=None)[0] for j, d in enumerate(design)])
        prediction = np.column_stack([d @ coef[j] for j, d in enumerate(design)])
        error = np.linalg.norm(prediction - raw, axis=1)
        scale = max(2.5 * np.median(error), 1e-12)
        weight = 1 / (1 + (error / scale) ** 2)
    return coef


def _invert_rosette(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4) - coef[:, 0]
    linear, cubic, pair = coef[:, 1:5], coef[:, 5], coef[:, 6]
    x = np.linalg.solve(linear, target.T).T
    j = np.arange(4)
    k = (j + 1) % 4

    def residual(v):
        return v @ linear.T + cubic * v ** 3 + pair * v * v[:, k] - target

    err = residual(x)
    for _ in range(30):
        norm = np.max(np.abs(err), axis=1)
        active = norm > 1e-12 * (1 + np.max(np.abs(target), axis=1))
        if not np.any(active):
            return x.reshape(raw.shape)
        xa = x[active]
        jac = np.broadcast_to(linear, (len(xa), 4, 4)).copy()
        jac[:, j, j] += 3 * cubic * xa ** 2 + pair * xa[:, k]
        jac[:, j, k] += pair * xa
        step = np.zeros_like(x)
        step[active] = np.linalg.solve(jac, err[active, :, None])[:, :, 0]
        trial = x - step
        trial_err = residual(trial)
        for _ in range(20):
            worse = active & (np.max(np.abs(trial_err), axis=1) >= norm)
            if not np.any(worse):
                break
            step[worse] *= .5
            trial = x - step
            trial_err = residual(trial)
        x, err = trial, trial_err
    raise RuntimeError('Rosette inverse did not converge')


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coef = _fit_rosette(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    calibrated = dict(study, y_obs_raw=_invert_rosette(study['y_obs_raw'], coef).ravel())
    banks = sweeps['single_bank_sweeps']
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(banks, raw_readout=banks['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert_rosette(pulse['y_pulse_raw'], coef))
            for pulse in pulses]
    return calibrated
