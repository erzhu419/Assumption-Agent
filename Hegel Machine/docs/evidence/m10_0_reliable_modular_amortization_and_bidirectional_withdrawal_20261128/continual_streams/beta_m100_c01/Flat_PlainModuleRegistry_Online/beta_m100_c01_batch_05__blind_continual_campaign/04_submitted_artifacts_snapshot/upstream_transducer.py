"""Fit the rosette's forward response from paired sweeps and invert it."""

import numpy as np


def _design(x, channel, sources):
    z = x[:, channel]
    t = x[:, sources]
    return np.column_stack((np.ones(len(x)), z, np.sin(2.5 * z), t, np.sin(t)**2))


def _predict(x, offset, linear, diagonal, even):
    return offset + x @ linear.T + np.sin(2.5 * x) * diagonal + np.sin(x)**2 @ even.T


def _fit(x, y, paired):
    sources = [[j ^ 2] if paired else [k for k in range(4) if k != j]
               for j in range(4)]
    designs = [_design(x, j, src) for j, src in enumerate(sources)]
    weights = np.ones(len(x))

    def fit(weights):
        return np.array([np.linalg.lstsq(a * weights[:, None], y[:, j] * weights,
                                         rcond=None)[0] for j, a in enumerate(designs)])

    def residual(b):
        return np.column_stack([y[:, j] - a @ b[j] for j, a in enumerate(designs)])

    # Weight complete rows: a spike in one channel can coincide with others.
    for _ in range(24):
        b = fit(weights)
        err = np.linalg.norm(residual(b), axis=1)
        scale = max(np.median(err), 1e-12)
        updated = 1.0 / np.hypot(1.0, err / (2.0 * scale))
        if np.max(np.abs(updated - weights)) < 1e-4:
            break
        weights = updated
    inliers = err <= 6 * scale
    b = fit(inliers.astype(float))
    err = residual(b)[inliers]
    offset, diagonal = b[:, 0], b[:, 2]
    linear = np.diag(b[:, 1])
    even = np.zeros((4, 4))
    for j, src in enumerate(sources):
        linear[j, src] = b[j, 3:3 + len(src)]
        even[j, src] = b[j, 3 + len(src):]
    return (offset, linear, diagonal, even), np.sqrt(np.mean(err**2))


def _fit_rosette(x, y):
    model, paired_error = _fit(x, y, paired=True)
    # Use the observed pair symmetry unless the data resolve extra couplings.
    full, full_error = _fit(x, y, paired=False)
    return full if full_error < 0.6 * paired_error else model


def _invert(raw, model):
    shape = np.shape(raw)
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    offset, linear, diagonal, even = model
    axis = np.arange(4)
    tangent = linear + np.diag(2.5 * diagonal)
    x = np.linalg.solve(tangent, (y - offset).T).T
    tol = 1e-12 * (1.0 + np.linalg.norm(y, axis=1))
    for _ in range(40):
        residual = _predict(x, *model) - y
        norm = np.linalg.norm(residual, axis=1)
        active = norm > tol
        if not active.any():
            break
        z = x[active]
        jac = linear + even * np.sin(2.0 * z)[:, None, :]
        jac[:, axis, axis] += (2.5 * diagonal) * np.cos(2.5 * z)
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        dest = z - step
        base = norm[active]
        target = y[active]
        for _ in range(16):
            retry = np.linalg.norm(_predict(dest, *model) - target, axis=1) >= base
            if not retry.any():
                break
            step[retry] *= 0.5
            dest[retry] = z[retry] - step[retry]
        x[active] = dest
    return x.reshape(shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated study, leaving the supplied study unchanged."""
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    model = _fit_rosette(ref, raw)
    single = sweeps['single_bank_sweeps']
    calibrated = dict(study, y_obs_raw=_invert(study['y_obs_raw'], model))
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(single, raw_readout=single['ref_signal'].copy()),
    )
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], model))
            if pulse is not None and pulse.get('y_pulse_raw') is not None else pulse
            for pulse in pulses
        ]
    return calibrated
