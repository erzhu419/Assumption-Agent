"""Robust forward calibration and inversion of the four-channel rosette."""
import numpy as np

_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4
_PREV = (_CHANNELS - 1) % 4


def _basis(ref):
    nxt, prv = ref[:, _NEXT], ref[:, _PREV]
    return np.stack((np.ones_like(ref), ref, nxt, prv, ref * nxt, ref**3), axis=-1)


def _fit(sweeps):
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    basis = _basis(ref)
    # Screen gross EMI before fitting, then trim by joint forward residual.
    magnitude = np.linalg.norm(raw - np.median(raw, axis=0), axis=1)
    keep = magnitude <= 8 * np.median(magnitude)
    for _ in range(4):
        coef = np.array([np.linalg.lstsq(basis[keep, j], raw[keep, j], rcond=None)[0]
                         for j in _CHANNELS])
        error = np.linalg.norm(raw - np.einsum('nck,ck->nc', basis, coef), axis=1)
        refined = error <= max(1e-10, 6 * np.median(error))
        if np.array_equal(refined, keep):
            break
        keep = refined

    bank = sweeps.get('single_bank_sweeps')
    if bank is None:
        return coef
    channel = np.asarray(bank['bank_id'], dtype=int)
    signal = np.asarray(bank['ref_signal'], dtype=float)
    readout = np.asarray(bank['raw_readout'], dtype=float)
    axis = np.column_stack((np.ones_like(signal), signal, np.zeros_like(signal),
                            np.zeros_like(signal), np.zeros_like(signal), signal**3))
    axis_error = np.abs(readout - np.einsum('nk,nk->n', axis, coef[channel]))
    good = axis_error <= max(1e-10, 6 * np.median(error), 6 * np.median(axis_error))
    for j in _CHANNELS:
        pick = good & (channel == j)
        coef[j] = np.linalg.lstsq(np.vstack((basis[keep, j], axis[pick])),
                                 np.r_[raw[keep, j], readout[pick]], rcond=None)[0]
    return coef


def _invert(raw, coef):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    const, own, nxt, prv, cross, cubic = coef.T
    linear = np.zeros((4, 4))
    linear[_CHANNELS, _CHANNELS] = own
    linear[_CHANNELS, _NEXT] = nxt
    linear[_CHANNELS, _PREV] = prv
    ref = np.linalg.solve(linear, (target - const).T).T

    def residual(x):
        return x @ linear.T + const + cross * x * x[:, _NEXT] + cubic * x**3 - target

    for _ in range(30):
        r = residual(ref)
        err = np.max(np.abs(r), axis=1)
        if not np.any(err > 1e-12):
            break
        jac = np.broadcast_to(linear, (len(ref), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += cross * ref[:, _NEXT] + 3 * cubic * ref**2
        jac[:, _CHANNELS, _NEXT] += cross * ref
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = ref - step
        for _ in range(12):
            retry = (err > 1e-12) & (np.max(np.abs(residual(trial)), axis=1) >= err)
            if not retry.any():
                break
            step[retry] *= 0.5
            trial[retry] = ref[retry] - step[retry]
        ref = trial
    return ref.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated copy of the study in reference coordinates."""
    sweeps = study['sensor_calibration_sweeps']
    coef = _fit(sweeps)
    out = dict(study)
    out['y_obs_raw'] = _invert(study['y_obs_raw'], coef)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coef))
                                          for pulse in pulses]
    normalized = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    bank = sweeps.get('single_bank_sweeps')
    if bank is not None:
        normalized['single_bank_sweeps'] = dict(bank, raw_readout=bank['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = normalized
    return out
