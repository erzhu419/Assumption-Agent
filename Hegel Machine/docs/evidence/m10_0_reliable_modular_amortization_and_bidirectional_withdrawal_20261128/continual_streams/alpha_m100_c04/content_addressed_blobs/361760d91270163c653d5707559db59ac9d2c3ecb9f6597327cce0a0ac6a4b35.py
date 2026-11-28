"""Fit and invert the coupled four-channel chamber bridge."""
import numpy as np
from scipy.optimize import least_squares


_OPPOSITE = np.array([2, 3, 0, 1])


def _basis(own, opposite, frequencies):
    sine, cosine = frequencies
    return np.stack((np.ones_like(own), own, opposite, np.sin(sine * own),
                     np.cos(cosine * opposite) - 1), axis=-1)


def _fit(calibration):
    loads = np.asarray(calibration['true_channel_loads'], dtype=float)
    raw = np.asarray(calibration['bridge_raw'], dtype=float)
    sweeps = calibration['single_channel_sweeps']
    channel = np.asarray(sweeps['lab_id'])
    stress = np.asarray(sweeps['true_stress'], dtype=float)
    command = np.asarray(sweeps['cmd_stress'], dtype=float)
    n_bridge = len(loads)
    observations = [np.r_[raw[:, j], command[channel == j]] for j in range(4)]
    axes = [np.r_[loads[:, j], stress[channel == j]] for j in range(4)]
    cross = [np.r_[loads[:, k], np.zeros(np.count_nonzero(channel == j))]
             for j, k in enumerate(_OPPOSITE)]

    # A bounded-influence pilot keeps gross ADC spikes out of the frequency fit.
    def pilot_weight(amplitude):
        limit = max(4 * np.median(amplitude), 1.0)
        return np.minimum(1, (limit / np.maximum(amplitude, limit)) ** 4)

    bridge_amp = np.max(np.abs(raw), axis=1)
    weights = [np.r_[pilot_weight(bridge_amp), pilot_weight(np.abs(command[channel == j]))]
               for j in range(4)]

    def profile(shape, weights):
        sine, cosine, cross_fraction, sine_fraction, cos_fraction = shape
        coefficients, errors = [], []
        for t, u, y, w in zip(axes, cross, observations, weights):
            g = (t + cross_fraction * u + sine_fraction * np.sin(sine * t)
                 + cos_fraction * (np.cos(cosine * u) - 1))
            g_center = g - np.average(g, weights=w)
            y_mean = np.average(y, weights=w)
            gain = np.dot(w * g_center, y - y_mean) / np.dot(w, g_center**2)
            bias = y_mean - gain * np.average(g, weights=w)
            coefficients.append((bias, gain, gain * cross_fraction,
                                 gain * sine_fraction, gain * cos_fraction))
            errors.append(bias + gain * g - y)
        return np.asarray(coefficients), errors

    # All four bridges share a normalized response; offsets and gains are local.
    shape = np.array([2.5, 2.0, 0.22, 0.25, 0.06])
    for step in range(4):
        def objective(p):
            _, errors = profile(p, weights)
            return np.concatenate([e * np.sqrt(w) for e, w in zip(errors, weights)])

        shape = least_squares(objective, shape,
                              bounds=([2.0, 1.5, -np.inf, -np.inf, -np.inf],
                                      [3.0, 2.5, np.inf, np.inf, np.inf])).x
        coefficients, errors = profile(shape, weights)
        if step < 3:
            e = np.concatenate(errors)
            scale = max(1.4826 * np.median(np.abs(e - np.median(e))), 1e-7)
            # Share the rejection weight across all four outputs of a bridge row.
            amplitude = np.max(np.abs([e[:n_bridge] for e in errors]), axis=0)
            cutoff = 8 * scale
            weights = [1 / (1 + (np.r_[amplitude, np.abs(e[n_bridge:])] / cutoff) ** 2)
                       for e in errors]
    return coefficients, shape[:2]


def _forward(loads, coefficients, frequencies):
    return np.sum(_basis(loads, loads[..., _OPPOSITE], frequencies) * coefficients, axis=-1)


def _invert(values, coefficients, frequencies):
    raw = np.asarray(values, dtype=float)
    target = raw.reshape(-1, 4)
    bias, linear, cross, sine_amp, cos_amp = coefficients.T
    sine, cosine = frequencies
    loads = (target - bias) / linear
    for _ in range(60):
        error = _forward(loads, coefficients, frequencies) - target
        if np.max(np.abs(error), initial=0.0) < 1e-12:
            break
        diagonal = linear + sine_amp * sine * np.cos(sine * loads)
        off_diag = cross - cos_amp * cosine * np.sin(cosine * loads[:, _OPPOSITE])
        determinant = diagonal * diagonal[:, _OPPOSITE] - off_diag * off_diag[:, _OPPOSITE]
        step = (diagonal[:, _OPPOSITE] * error - off_diag * error[:, _OPPOSITE]) / determinant
        current = np.max(np.abs(error), axis=1)
        trial = loads - step
        for _ in range(20):
            worse = (np.max(np.abs(_forward(trial, coefficients, frequencies) - target), axis=1)
                     > current) & (current > 1e-12)
            if not np.any(worse):
                break
            trial[worse] = (loads[worse] + trial[worse]) * 0.5
        loads = trial
    return loads.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in canonical load coordinates without editing the input."""
    calibration = dataset['calibration_load_cells']
    coefficients, frequencies = _fit(calibration)

    def convert(block):
        return {**block, **{key: _invert(block[key], coefficients, frequencies)
                            for key in ('past_channel_raw', 'hold_channel_raw')}}

    sweeps = calibration['single_channel_sweeps']
    result = {**dataset, 'train': convert(dataset['train']), 'test': convert(dataset['test']),
              'calibration_load_cells': {
                  **calibration, 'bridge_raw': calibration['true_channel_loads'].copy(),
                  'single_channel_sweeps': {**sweeps, 'cmd_stress': sweeps['true_stress'].copy()},
              }}
    trials = dataset.get('chamber_verification_trials')
    if trials is not None:
        result['chamber_verification_trials'] = convert(trials)
    return result
