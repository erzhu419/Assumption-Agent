"""Robust paired-harmonic bridge calibration in canonical load coordinates."""

import numpy as np
from scipy.optimize import least_squares


_PARTNER = np.array([2, 3, 0, 1])
_INITIAL_FREQ = np.array([2.5, 2.0])


def _basis(x, freq):
    z = x[:, _PARTNER]
    return np.stack((np.ones_like(x), x, np.sin(freq[0] * x),
                     z, np.cos(freq[1] * z) - 1), axis=-1)


def _fit(cal):
    x = np.asarray(cal['true_channel_loads'], dtype=float)
    y = np.asarray(cal['bridge_raw'], dtype=float)

    def project(freq, weights):
        a = _basis(x, freq)
        w = np.sqrt(weights)
        beta = np.stack([np.linalg.lstsq(a[:, j] * w[:, None], y[:, j] * w,
                                        rcond=None)[0] for j in range(4)])
        return beta, np.einsum('njk,jk->nj', a, beta) - y

    # Downweight entire spike rows, using known loads as fixed predictors.
    delta = y - x
    distance = np.linalg.norm(delta - np.median(delta, axis=0), axis=1)
    for _ in range(8):
        scale = max(1e-12, 2 * np.median(distance))
        _, err = project(_INITIAL_FREQ, 1 / (1 + (distance / scale)**2))
        distance = np.linalg.norm(err, axis=1)
    weights = (distance <= max(1e-10, 8 * np.median(distance))).astype(float)
    # The channels share the same normalized response; gain and offset remain
    # channel-specific. Profile them out of the nonlinear fit.
    good = weights.astype(bool)
    initial, _ = project(_INITIAL_FREQ, weights)
    seed = np.r_[_INITIAL_FREQ, np.median(initial[:, 2:] / initial[:, 1, None], axis=0)]
    truth, raw = x[good], y[good]
    partner = truth[:, _PARTNER]
    raw_mean = raw.mean(axis=0)
    raw_centered = raw - raw_mean

    def shared(params):
        h, k, sine, cross, cosine = params
        shape = (truth + sine * np.sin(h * truth) + cross * partner
                 + cosine * (np.cos(k * partner) - 1))
        center = shape.mean(axis=0)
        centered = shape - center
        gain = np.sum(centered * raw_centered, axis=0) / np.sum(centered**2, axis=0)
        offset = raw_mean - gain * center
        return shape * gain + offset - raw, gain, offset

    params = least_squares(lambda p: shared(p)[0].ravel(), seed,
                           gtol=1e-12, ftol=1e-10, xtol=1e-10).x
    _, gain, offset = shared(params)
    return params[:2], np.column_stack((offset, gain, gain[:, None] * params[2:]))


def _forward(x, freq, beta):
    return np.einsum('njk,jk->nj', _basis(x, freq), beta)


def _inverse(raw, freq, beta):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    offset, gain, sine, cross, cosine = beta.T
    x = (y - offset) / gain
    h, k = freq
    for _ in range(24):
        r = _forward(x, freq, beta) - y
        norm = np.max(np.abs(r), axis=1)
        active = norm > 2e-12
        if not np.any(active):
            return x.reshape(raw.shape)
        q, r = x[active], r[active]
        target = y[active]
        diagonal = gain + h * sine * np.cos(h * q)
        coupled = cross - k * cosine * np.sin(k * q[:, _PARTNER])
        det = diagonal * diagonal[:, _PARTNER] - coupled * coupled[:, _PARTNER]
        step = (diagonal[:, _PARTNER] * r - coupled * r[:, _PARTNER]) / det
        # Backtrack independently for each reading, especially near shallow slopes.
        scale = np.ones((len(q), 1))
        trial = q - scale * step
        for _ in range(20):
            worse = np.max(np.abs(_forward(trial, freq, beta) - target), axis=1) >= norm[active]
            if not np.any(worse):
                break
            scale[worse] *= 0.5
            trial[worse] = q[worse] - scale[worse] * step[worse]
        x[active] = trial
    raise RuntimeError('Bridge inverse did not converge')


def _calibrate_part(part, freq, beta):
    if part is None:
        return None
    if isinstance(part, (list, tuple)):
        return type(part)(_calibrate_part(trial, freq, beta) for trial in part)
    out = part.copy()
    for key in ('past_channel_raw', 'hold_channel_raw'):
        if key in part:
            out[key] = _inverse(part[key], freq, beta)
    return out


def calibrate_upstream(dataset: dict) -> dict:
    """Return a calibrated copy without changing the raw dataset."""
    cal = dataset['calibration_load_cells']
    freq, beta = _fit(cal)
    sweeps = cal['single_channel_sweeps']
    out = dataset.copy()
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    for name in ('train', 'test', 'chamber_verification_trials'):
        if name in dataset:
            out[name] = _calibrate_part(dataset[name], freq, beta)
    return out
