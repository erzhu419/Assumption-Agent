"""Robust calibration for the four-channel load bridge."""
import numpy as np


_PARTNER = np.array([2, 3, 0, 1])


def _design(loads):
    partner = loads[:, _PARTNER]
    return np.stack((np.ones_like(loads), loads + .25*np.sin(2.5*loads),
                     partner, 1 - np.cos(2*partner)), axis=-1)


def _fit(loads, raw):
    # One weight per calibration row: a spike can affect several outputs.
    design = _design(loads)
    weight = np.ones(len(loads))
    for _ in range(12):
        beta = np.array([np.linalg.lstsq(design[:, j]*weight[:, None],
                                        raw[:, j]*weight, rcond=None)[0]
                         for j in range(4)]).T
        error = np.max(np.abs(np.sum(design*beta.T, axis=-1) - raw), axis=1)
        scale = max(4*np.median(error), 1e-10)
        weight = 1/np.hypot(1, error/scale)
    # Refit clean rows without shrinking legitimate observations.
    keep = error <= scale*2
    return np.array([np.linalg.lstsq(design[keep, j], raw[keep, j], rcond=None)[0]
                     for j in range(4)]).T


def _inverse(raw, beta):
    raw = np.asarray(raw, dtype=float)
    offset, gain, linear, cosine = beta
    target = (raw.reshape(-1, 4) - offset)/gain
    linear, cosine = linear/gain, cosine/gain
    loads = target.copy()

    def residual(x):
        p = x[:, _PARTNER]
        return x + .25*np.sin(2.5*x) + linear*p + cosine*(1 - np.cos(2*p)) - target

    r = residual(loads)
    # The fitted bridge is diagonally dominant. Solve its two coupled
    # 2-by-2 Newton systems in parallel and backtrack per measurement.
    for _ in range(40):
        norm = np.max(np.abs(r), axis=1, keepdims=True)
        if np.all(norm <= 1e-11):
            return loads.reshape(raw.shape)
        a = 1 + .625*np.cos(2.5*loads)
        b = linear + 2*cosine*np.sin(2*loads[:, _PARTNER])
        step = (a[:, _PARTNER]*r - b*r[:, _PARTNER]) / (
            a*a[:, _PARTNER] - b*b[:, _PARTNER])
        step = np.where(norm > 1e-11, step, 0.)
        trial = loads - step
        rt = residual(trial)
        for _ in range(20):
            retry = np.max(np.abs(rt), axis=1, keepdims=True) > norm
            if not np.any(retry):
                break
            step = np.where(retry, step*.5, step)
            trial = loads - step
            rt = residual(trial)
        loads, r = trial, rt
    raise RuntimeError('load bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Convert the bridge readings to load coordinates without mutating inputs."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    result = dict(dataset)
    if not np.array_equal(raw, loads):
        beta = _fit(loads, raw)

        def convert(record):
            out = dict(record)
            for key in ('past_channel_raw', 'hold_channel_raw'):
                if key in record:
                    out[key] = _inverse(record[key], beta)
            return out

        for key in ('train', 'test', 'chamber_verification_trials'):
            record = dataset.get(key)
            if isinstance(record, dict):
                result[key] = convert(record)
            elif record is not None:
                result[key] = type(record)(convert(trial) for trial in record)
    result['calibration_load_cells'] = dict(cal, bridge_raw=cal['true_channel_loads'].copy())
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells']['single_channel_sweeps'] = dict(
        sweeps, cmd_stress=sweeps['true_stress'].copy())
    return result
