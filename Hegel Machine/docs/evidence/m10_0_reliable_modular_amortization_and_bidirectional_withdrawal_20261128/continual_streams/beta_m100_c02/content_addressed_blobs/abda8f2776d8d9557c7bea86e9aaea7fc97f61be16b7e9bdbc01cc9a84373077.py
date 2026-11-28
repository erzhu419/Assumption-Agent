"""Calibrate the four-channel rosette into the solver's reference coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _design(x, frequency, channel):
    odd, even = frequency
    other = np.arange(4) != channel
    return np.column_stack((np.ones(len(x)), x,
                            np.sin(odd * x[:, channel]),
                            1 - np.cos(even * x[:, other])))


def _fit(ref, raw, single):
    # Estimate an initial residual scale without letting gross EMI rows set it.
    a = np.column_stack((np.ones(len(ref)), ref))
    coef = np.linalg.lstsq(a, raw, rcond=None)[0]
    scale = max(np.median(np.linalg.norm(a @ coef - raw, axis=1)), 1e-10)
    for _ in range(12):
        r = np.linalg.norm(a @ coef - raw, axis=1)
        w = 1 / np.hypot(1, r / scale)
        coef = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
    r = np.linalg.norm(a @ coef - raw, axis=1)
    good = r <= 7 * max(np.median(r), 1e-10)

    axes = []
    bank = np.asarray(single['bank_id'])
    for j in range(4):
        q = bank == j
        signal = np.asarray(single['ref_signal'])[q]
        readout = np.asarray(single['raw_readout'])[q]
        x = np.zeros((len(signal), 4))
        x[:, j] = signal
        dev = np.abs(readout - np.median(readout))
        accepted = dev <= 6 * max(np.median(dev), 1e-10)
        axes.append((x[accepted], readout[accepted]))

    frequency = np.array([2.5, 2.0])
    for _ in range(3):
        samples = [(np.vstack((ref[good], x)), np.r_[raw[good, j], y])
                   for j, (x, y) in enumerate(axes)]

        def profile(freq, coefficients=False):
            result = []
            for j, (x, y) in enumerate(samples):
                a = _design(x, freq, j)
                beta = np.linalg.lstsq(a, y, rcond=None)[0]
                result.append(beta if coefficients else a @ beta - y)
            return np.array(result) if coefficients else np.concatenate(result)

        frequency = least_squares(profile, frequency, bounds=(0.3, 6.0),
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x
        coef = profile(frequency, coefficients=True)
        r = np.linalg.norm(_forward(ref, frequency, coef) - raw, axis=1)
        updated = r <= 8 * max(np.median(r), 1e-10)
        if np.array_equal(updated, good):
            break
        good = updated
    return frequency, coef


def _forward(x, frequency, coef):
    return np.column_stack([_design(x, frequency, j) @ coef[j] for j in range(4)])


def _inverse(raw, frequency, coef):
    values = np.asarray(raw, dtype=float)
    target = values.reshape(-1, 4)
    odd, even = frequency
    intercept = coef[:, 0]
    linear = coef[:, 1:5]
    sinusoid = coef[:, 5]
    coupling = np.zeros((4, 4))
    off = ~np.eye(4, dtype=bool)
    coupling[off] = coef[:, 6:].ravel()
    x = np.linalg.solve(linear + np.diag(odd * sinusoid), (target - intercept).T).T
    diag = np.arange(4)
    for _ in range(30):
        error = _forward(x, frequency, coef) - target
        norms = np.linalg.norm(error, axis=1)
        active = np.flatnonzero(norms > 1e-11)
        if not len(active):
            break
        xa = x[active]
        jac = linear + coupling * (even * np.sin(even * xa[:, None, :]))
        jac[:, diag, diag] += odd * sinusoid * np.cos(odd * xa)
        delta = np.linalg.solve(jac, error[active, :, None])[..., 0]
        # A full Newton step is fast near zero; backtrack near saturation.
        for power in range(16):
            trial = xa - delta * (0.5 ** power)
            new_norm = np.linalg.norm(_forward(trial, frequency, coef) - target[active], axis=1)
            improved = new_norm < norms[active]
            x[active[improved]] = trial[improved]
            remains = ~improved
            active, xa, delta = active[remains], xa[remains], delta[remains]
            if not len(active):
                break
    return x.reshape(values.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return calibrated observations and pulses, leaving the input unchanged."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    single = sweeps['single_bank_sweeps']
    model = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float), single)
    result = dict(study)
    result['y_obs_raw'] = _inverse(study['y_obs_raw'], *model)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(p, y_pulse_raw=_inverse(p['y_pulse_raw'], *model))
            if p.get('y_pulse_raw') is not None else dict(p) for p in pulses]
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(),
        single_bank_sweeps=dict(single, raw_readout=np.asarray(single['ref_signal']).copy()))
    return result
