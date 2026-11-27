"""Calibrate the four-channel rosette before downstream source inversion."""
import numpy as np


_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4


def _design(x, channel):
    own = x[:, channel]
    return np.column_stack((np.ones(len(x)), x, own * x[:, _NEXT[channel]], own**3))


def _fit(ref, raw):
    # The linear mixing is unrestricted. Each output also has its own cubic
    # response and a quadratic interaction with its next rosette channel.
    design = [_design(ref, channel) for channel in _CHANNELS]
    weight = np.ones(len(ref))
    for _ in range(20):
        coeff = np.column_stack([
            np.linalg.lstsq(a * weight[:, None], raw[:, j] * weight, rcond=None)[0]
            for j, a in enumerate(design)
        ])
        predicted = np.column_stack([a @ coeff[:, j] for j, a in enumerate(design)])
        error = np.linalg.norm(predicted - raw, axis=1)
        scale = max(2.5 * np.median(error), 1e-12)
        # A common row weight keeps an EMI event from biasing any channel.
        weight = 1 / np.hypot(1, error / scale)
    return coeff


def _forward(x, coeff):
    return coeff[0] + x @ coeff[1:5] + coeff[5] * x * x[:, _NEXT] + coeff[6] * x**3


def _invert(raw, coeff):
    shape = np.shape(raw)
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    x = np.linalg.solve(coeff[1:5].T, (y - coeff[0]).T).T.copy()
    for _ in range(30):
        residual = _forward(x, coeff) - y
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-12 * (1 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            break
        z = x[active]
        jac = np.broadcast_to(coeff[1:5].T, (len(z), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += coeff[5] * z[:, _NEXT] + 3 * coeff[6] * z**2
        jac[:, _CHANNELS, _NEXT] += coeff[5] * z
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        target, old = y[active], error[active]
        trial = z - step
        for _ in range(16):
            worse = np.linalg.norm(_forward(trial, coeff) - target, axis=1) >= old
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    return x.reshape(shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a copy with observations and reference pulses in canonical units."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    coeff = _fit(ref, np.asarray(sweeps['raw_rosette'], dtype=float))
    single = sweeps['single_bank_sweeps']
    result = dict(study)
    result['y_obs_raw'] = _invert(study['y_obs_raw'], coeff)
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        result['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], coeff)) for pulse in pulses
        ]
    result['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(single, raw_readout=single['ref_signal'].copy()),
    )
    return result
