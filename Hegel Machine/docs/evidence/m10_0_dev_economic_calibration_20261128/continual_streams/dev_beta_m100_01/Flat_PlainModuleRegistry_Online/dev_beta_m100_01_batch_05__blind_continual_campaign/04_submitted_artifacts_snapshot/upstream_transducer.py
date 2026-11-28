"""Fit and invert the coupled four-bank rosette response."""
import numpy as np
from scipy.optimize import least_squares

_OPPOSITE = np.array([2, 3, 0, 1])


def _response(x, p):
    bias, gain = p[:4], p[4:8]
    slope, amp, freq, cross, even, cross_freq = p[8:]
    q = x[:, _OPPOSITE]
    return bias + gain * (slope * x + amp * np.sin(freq * x) +
                          cross * q + even * (1 - np.cos(cross_freq * q)))


def _fit(ref, raw, bank):
    j = np.asarray(bank['bank_id'], dtype=int)
    axis = np.asarray(bank['ref_signal'], dtype=float)
    readout = np.asarray(bank['raw_readout'], dtype=float)
    # Gross calibration spikes are whole-row contamination for the rosette.
    def gross_mask(v):
        med = np.median(v)
        return v <= med + 10 * max(np.median(np.abs(v - med)), med * .1, 1e-5)

    good = gross_mask(np.max(np.abs(raw), axis=1))
    axis_good = gross_mask(np.abs(readout))
    center = np.median(raw[good], axis=0)
    ranges = np.quantile(raw[good], .9, axis=0) - np.quantile(raw[good], .1, axis=0)
    start = np.r_[center, 1., np.clip(ranges[1:] / max(ranges[0], 1e-8), .1, 10.),
                  1.2, .3, 2.5, .25, -.07, 2.]
    lo = np.r_[[-np.inf]*4, 1., [.025]*3, 1e-8, -np.inf, .1, -np.inf, -np.inf, .1]
    hi = np.r_[[np.inf]*4, 1., [40.]*3, np.inf, np.inf, 7., np.inf, np.inf, 7.]

    def unpack(v):
        return np.insert(v, 4, 1.)  # Fix the common gain/shape scaling ambiguity.

    def errors(p):
        slope, amp, freq = p[8:11]
        isolated = p[j] + p[4+j] * (slope * axis + amp * np.sin(freq * axis))
        return _response(ref, p) - raw, isolated - readout

    def residual(v):
        rows, single = errors(unpack(v))
        return np.r_[rows[good].ravel(), single[axis_good]]

    v = np.delete(start, 4)
    bounds = (np.delete(lo, 4), np.delete(hi, 4))
    p = least_squares(residual, v, bounds=bounds, loss='soft_l1', f_scale=.03,
                      max_nfev=200).x
    # Refine after a residual-based EMI screen; allow for ordinary sweep noise.
    rows, single = errors(unpack(p))
    r = np.linalg.norm(rows, axis=1)
    good &= r <= max(8 * np.median(r[good]), 1e-4)
    axis_good &= np.abs(single) <= max(8 * np.median(np.abs(single[axis_good])), 1e-4)
    scale = max(1e-5, 2 * np.median(np.abs(residual(p))))
    return unpack(least_squares(residual, p, bounds=bounds, loss='soft_l1',
                                f_scale=scale, max_nfev=150).x)


def _inverse(raw, p):
    raw = np.asarray(raw, dtype=float)
    z = (raw.reshape(-1, 4) - p[:4]) / p[4:8]
    slope, amp, freq, cross, even, cross_freq = p[8:]
    x = z / slope
    for a, b in ((0, 2), (1, 3)):
        x[:, a] = (slope*z[:, a] - cross*z[:, b]) / (slope*slope - cross*cross)
        x[:, b] = (slope*z[:, b] - cross*z[:, a]) / (slope*slope - cross*cross)

    def residual(x):
        q = x[:, _OPPOSITE]
        return slope*x + amp*np.sin(freq*x) + cross*q + even*(1-np.cos(cross_freq*q)) - z

    r = residual(x)
    for _ in range(40):
        err = np.max(np.abs(r), axis=1)
        if not np.any(err > 2e-12):
            break
        d = slope + amp*freq*np.cos(freq*x)
        e = cross + even*cross_freq*np.sin(cross_freq*x[:, _OPPOSITE])
        det = d*d[:, _OPPOSITE] - e*e[:, _OPPOSITE]
        step = (d[:, _OPPOSITE]*r - e*r[:, _OPPOSITE]) / det
        trial = x - step
        new_r = residual(trial)
        for _ in range(16):
            bad = np.max(np.abs(new_r), axis=1) >= err
            if not np.any(bad & (err > 2e-12)):
                break
            step[bad] *= .5
            trial[bad] = x[bad] - step[bad]
            new_r = residual(trial)
        accept = np.max(np.abs(new_r), axis=1) < err
        x[accept], r[accept] = trial[accept], new_r[accept]
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    single = sweeps['single_bank_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    out = dict(study)
    out['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=ref.copy(), single_bank_sweeps=dict(
            single, raw_readout=np.asarray(single['ref_signal']).copy()))
    if np.array_equal(ref, raw):
        return out
    p = _fit(ref, raw, single)
    out['y_obs_raw'] = _inverse(study['y_obs_raw'], p)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], p))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses]
    return out
