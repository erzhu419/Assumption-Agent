"""Reconstruct two compact sources from a thermoacoustic ring scan."""

import numpy as np
from scipy.ndimage import correlate1d
from scipy.optimize import minimize, minimize_scalar


_SPEED_LIMITS = (1435.0, 1545.0)


def _pulse(u):
    return -u * np.exp(-0.5 * u * u)


def _matched(traces, width):
    lag = np.arange(-int(np.ceil(6 * width)), int(np.ceil(6 * width)) + 1)
    template = _pulse(lag / width)
    return correlate1d(traces, template, axis=1, mode='constant') / np.sqrt(template @ template)


def _sample(traces, arrival):
    base = np.floor(arrival).astype(int)
    fraction = arrival - base
    k = np.arange(traces.shape[0])
    return traces[k, base] * (1 - fraction) + traces[k, base + 1] * fraction


def _pin_speed(matched, pin_range, samples_per_mm_times_speed):
    def score(speeds):
        arrival = samples_per_mm_times_speed * pin_range / np.asarray(speeds)[..., None]
        response = np.sort(np.abs(_sample(matched, arrival)), axis=-1)
        # Overlapped absorber arrivals can boost the pin response on a few channels.
        return response[..., 2:-4].mean(axis=-1)

    speeds = np.linspace(*_SPEED_LIMITS, 221)
    best = speeds[np.argmax(score(speeds))]
    bounds = (max(_SPEED_LIMITS[0], best - 1), min(_SPEED_LIMITS[1], best + 1))
    return float(minimize_scalar(lambda c: -score(c), bounds=bounds, method='bounded').x)


def reconstruct_two_absorbers_mm(scan: dict) -> dict:
    """Return sound speed (m/s) and the two absorber coordinates (mm)."""
    sensors = np.asarray(scan['sensor_xy_mm'], dtype=float)
    pin = np.asarray(scan['calib_pin_xy_mm'], dtype=float)
    traces = np.asarray(scan['signals'], dtype=float).T
    traces = traces - np.median(traces, axis=1, keepdims=True)
    time = np.arange(traces.shape[1])[None, :, None]
    scale = float(scan['fs_hz']) / 1000.0
    width0 = 0.22e-6 * float(scan['fs_hz'])
    pin_range = np.linalg.norm(sensors - pin, axis=1)
    speed0 = _pin_speed(_matched(traces, width0), pin_range, scale)
    speed_scale = 30.0
    speeds = tuple((c - speed0) / speed_scale for c in _SPEED_LIMITS)

    # Signed channel amplitudes are nuisance parameters. Profiling them out
    # accounts for gain/polarity without discarding the inverted channels.
    def profile(params, residual_only=False):
        speed = speed0 + speed_scale * params[0]
        width = width0 * params[1]
        xy = params[2:].reshape(-1, 2)
        displacement = xy[None, :, :] - sensors[:, None, :]
        direct_range = np.linalg.norm(displacement, axis=2)
        ranges = np.column_stack((pin_range, direct_range))
        centers = ranges * (scale / speed)
        u = (time - centers[:, None, :]) / width
        envelope = np.exp(-0.5 * u * u)
        templates = -u * envelope
        transpose = templates.transpose(0, 2, 1)
        gram = transpose @ templates
        ridge = 1e-5 * width0
        gram[:, np.arange(gram.shape[1]), np.arange(gram.shape[1])] += ridge
        amplitude = np.linalg.solve(gram, transpose @ traces[..., None])[..., 0]
        residual = traces - (templates @ amplitude[..., None])[..., 0]
        if residual_only:
            return residual

        loss = np.sum(residual**2) + ridge * np.sum(amplitude**2)
        # At the optimal amplitudes the profile gradient only needs derivatives
        # of the templates, not derivatives of the linear solves.
        common = (-2.0 / width) * residual[..., None] * amplitude[:, None, :] * (1 - u*u) * envelope
        dcenter = np.sum(common, axis=1)
        dspeed = np.sum(dcenter * centers) * (-speed_scale / speed)
        dwidth = np.sum(common * u) * width0
        dxy = np.einsum('ij,ijk->jk', dcenter[:, 1:] * scale / (speed * direct_range), displacement)
        return loss, np.r_[dspeed, dwidth, dxy.ravel()]

    axis = np.arange(-32.0, 32.01, 0.4)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= 32.0]
    grid_ranges = np.linalg.norm(grid[:, None, :] - sensors, axis=2)

    def seed(params):
        speed = speed0 + speed_scale * params[0]
        width = width0 * params[1]
        residual = profile(params, residual_only=True)
        arrival = grid_ranges * (scale / speed)
        response = _sample(_matched(residual, width), arrival)
        # Projecting out the pin removes some of a candidate's energy where
        # their pulses overlap. Normalize using the N-wave autocorrelation.
        lag = (arrival - pin_range * (scale / speed)) / width
        corr = (1 - 0.5 * lag**2) * np.exp(-0.25 * lag**2)
        score = np.sum(response**2 / np.maximum(1 - corr**2, 0.10), axis=1)
        return grid[np.argmax(score)]

    def refine(params, new_xy):
        initial = np.r_[params, new_xy]
        bounds = [speeds, (0.65, 1.55)] + [(-32., 32.)] * (initial.size - 2)
        return minimize(profile, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-11}).x

    params = np.array([0.0, 1.0])
    params = refine(params, seed(params))
    params = refine(params, seed(params))
    xy = sorted(params[2:].reshape(2, 2).tolist())
    return {'sound_speed_m_per_s': float(speed0 + speed_scale * params[0]),
            'absorbers_xy_mm': xy}
