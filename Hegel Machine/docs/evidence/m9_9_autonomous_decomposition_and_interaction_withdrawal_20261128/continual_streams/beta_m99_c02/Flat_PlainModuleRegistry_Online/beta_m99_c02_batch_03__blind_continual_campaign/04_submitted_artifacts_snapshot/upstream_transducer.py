"""Calibrate the coupled rosette before passing observations downstream."""
import numpy as np
from scipy.optimize import least_squares


def _design(x, j):
    own, nxt, prev = x[:, j], x[:, (j + 1) % 4], x[:, (j - 1) % 4]
    return np.column_stack((np.ones(len(x)), own, nxt, prev, own**3, own * nxt))


def _robust_fit(design, y):
    # Seed by least median residual, so a spike on a high-leverage axis
    # sweep cannot pull the fit that decides which readings are inliers.
    rng = np.random.default_rng(0)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    score = np.median(np.abs(design @ coef - y))
    for _ in range(64):
        ix = rng.choice(len(y), min(len(y), 10), replace=False)
        trial, _, rank, _ = np.linalg.lstsq(design[ix], y[ix], rcond=None)
        if rank != design.shape[1]:
            continue
        trial_score = np.median(np.abs(design @ trial - y))
        if trial_score < score:
            coef, score = trial, trial_score
    for _ in range(3):
        residual = np.abs(design @ coef - y)
        keep = residual <= max(6 * np.median(residual), 1e-10)
        coef = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return coef


def _fit_independent(sweeps):
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    solo = sweeps['single_bank_sweeps']
    bank = np.asarray(solo['bank_id'])
    signal = np.asarray(solo['ref_signal'], dtype=float)
    readout = np.asarray(solo['raw_readout'], dtype=float)
    coef = []
    for j in range(4):
        use = bank == j
        axis = np.zeros((use.sum(), 4))
        axis[:, j] = signal[use]
        design = _design(np.vstack((ref, axis)), j)
        coef.append(_robust_fit(design, np.r_[raw[:, j], readout[use]]))
    return np.array(coef)


def _fit(sweeps):
    base = _fit_independent(sweeps)
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    solo = sweeps['single_bank_sweeps']
    bank = np.asarray(solo['bank_id'])
    readout = np.asarray(solo['raw_readout'], dtype=float)
    idx = np.arange(len(bank))
    axis = np.zeros((len(bank), 4))
    axis[idx, bank] = solo['ref_signal']

    # Use the independent robust fit to reject contaminated readings before
    # estimating the chamber's common gain-normalized response.
    rr = np.abs(_forward(ref, base) - raw)
    sr = np.abs(_forward(axis, base)[idx, bank] - readout)
    gate = max(8 * np.median(np.r_[rr.ravel(), sr]), 1e-9)
    rm, sm = rr < gate, sr < gate
    seed = np.r_[base[:, 0], base[:, 1],
                 np.mean(base[:, 2:] / base[:, 1, None], axis=0)]

    def expand(p):
        return np.column_stack((p[:4], p[4:8], p[4:8, None] * p[None, 8:]))

    def residual(p):
        coef = expand(p)
        return np.r_[(_forward(ref, coef) - raw)[rm],
                     (_forward(axis, coef)[idx, bank] - readout)[sm]]

    return expand(least_squares(residual, seed, gtol=1e-12).x)


def _forward(x, coef):
    b, gain, next_gain, prev_gain, cubic, cross = coef.T
    nxt, prev = np.roll(x, -1, axis=-1), np.roll(x, 1, axis=-1)
    return b + gain*x + next_gain*nxt + prev_gain*prev + cubic*x**3 + cross*x*nxt


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    flat = raw.reshape(-1, 4)
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4
    b, gain, next_gain, prev_gain, cubic, cross = coef.T

    def jac(x):
        a = np.zeros((4, 4))
        a[j, j] = gain + 3*cubic*x**2 + cross*x[nxt]
        a[j, nxt] = next_gain + cross*x
        a[j, prev] = prev_gain
        return a

    start = np.linalg.solve(jac(np.zeros(4)), (flat - b).T).T
    out = np.empty_like(flat)
    for i, (y, x0) in enumerate(zip(flat, start)):
        out[i] = least_squares(lambda x: _forward(x, coef) - y, x0, jac=jac,
                               gtol=1e-12, ftol=1e-12, xtol=1e-12).x
    return out.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    coef = _fit(sweeps)
    solo = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_inverse(study['y_obs_raw'], coef))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(solo, raw_readout=solo['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_inverse(pulse['y_pulse_raw'], coef))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses]
    return calibrated
