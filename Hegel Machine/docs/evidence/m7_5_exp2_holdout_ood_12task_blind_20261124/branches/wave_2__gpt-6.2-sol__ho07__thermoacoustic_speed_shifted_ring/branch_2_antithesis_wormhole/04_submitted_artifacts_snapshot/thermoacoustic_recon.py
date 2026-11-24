"""Joint time-of-flight reconstruction for a thermoacoustic receiver ring."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import correlate1d
from scipy.optimize import least_squares, minimize, minimize_scalar


_SPEED_BOUNDS = (1435.0, 1545.0)


def _pulse(delta, width):
    u = delta / width
    return -u * np.exp(-0.5 * u * u)


def _matched(signals, width):
    radius = int(np.ceil(6 * width))
    template = _pulse(np.arange(-radius, radius + 1), width)
    trace = correlate1d(signals, template, axis=0, mode='constant')
    return CubicSpline(np.arange(len(signals)), trace).c, template @ template


def _sample(coeff, centers):
    """Interpolate one matched-filter value per candidate and receiver."""
    k = np.floor(centers).astype(int).clip(0, coeff.shape[1] - 1)
    c = coeff[:, k, np.arange(coeff.shape[2])]
    d = centers - k
    v = ((c[0] * d + c[1]) * d + c[2]) * d + c[3]
    return np.where((centers >= 0) & (centers <= coeff.shape[1]), v, 0.0)


def _project(atoms, obs):
    """Profile signed amplitudes independently at each receiver."""
    trans = atoms.swapaxes(1, 2)
    gram = trans @ atoms
    ridge = 1e-7
    gram += ridge * np.eye(gram.shape[-1])
    amps = np.linalg.solve(gram, trans @ obs[..., None])
    return (obs[..., None] - atoms @ amps)[..., 0], amps[..., 0]


def _seed(y, xy, pin_distance, factor, width):
    coeff, energy = _matched(y, width)
    def cost(speed):
        return -np.sum(_sample(coeff, pin_distance * factor / speed) ** 2)

    speeds = np.linspace(*_SPEED_BOUNDS, 111)
    k = int(np.argmin([cost(speed) for speed in speeds]))
    speed = minimize_scalar(cost, method='bounded', bounds=(
        speeds[max(0, k - 1)], speeds[min(k + 1, len(speeds) - 1)]
    )).x
    scale = factor / speed
    pin_t = pin_distance * scale
    pin_amp = _sample(coeff, pin_t) / energy
    polarity = np.where(pin_amp < 0, -1.0, 1.0)
    time = np.arange(len(y))[None, :, None]
    centers = pin_t[:, None]
    atoms = _pulse(time - centers[:, None, :], width)
    remaining = _project(atoms, y.T)[0].T

    axis = np.arange(-32.0, 32.01, 0.7)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = grid[np.sum(grid * grid, axis=1) < 32.0 ** 2]
    direct_t = np.linalg.norm(grid[:, None, :] - xy[None, :, :], axis=2) * scale
    def overlap(delta):
        # Normalized inner product of two shifted derivative-of-Gaussian atoms.
        v = (delta / (2.0 * width)) ** 2
        return (1.0 - 2.0 * v) * np.exp(-v)

    positions = []
    for _ in range(2):
        coeff, _ = _matched(remaining, width)
        gram = overlap(centers[:, :, None] - centers[:, None, :])
        inverse = np.linalg.inv(gram + 1e-6 * np.eye(gram.shape[-1]))

        def focus(arrival):
            cross = overlap(arrival[..., None] - centers)
            projected = 1.0 - np.einsum('...si,sij,...sj->...s', cross, inverse, cross)
            signed = np.maximum(_sample(coeff, arrival) * polarity, 0.0)
            return np.sum(signed ** 2 / np.maximum(projected, 0.10), axis=-1)

        scores = focus(direct_t)
        q = grid[int(np.argmax(scores))]
        # Sub-grid deflation prevents a strong packet's discretization error
        # from being selected instead of a weak second wavefront.
        def objective(point):
            arrival = np.linalg.norm(xy - point, axis=1) * scale
            return -float(focus(arrival))

        bounds = [(max(-32.0, a - 1.0), min(32.0, a + 1.0)) for a in q]
        point = minimize(objective, q, bounds=bounds, method='L-BFGS-B').x
        positions.append(point)
        centers = np.column_stack((centers, np.linalg.norm(xy - point, axis=1) * scale))
        atoms = _pulse(time - centers[:, None, :], width)
        remaining = _project(atoms, y.T)[0].T
    return np.r_[np.ravel(positions), speed, width]


def _fit(y, xy, pin_distance, factor, seed, *, pin_speed=None):
    def arrivals(params):
        points = params[:4].reshape(2, 2)
        d = np.linalg.norm(xy[:, None, :] - points[None, :, :], axis=2)
        return np.column_stack((d * (factor / params[4]),
                                pin_distance * (factor / (params[4] if pin_speed is None else pin_speed))))

    # Window around the three predicted packets, keeping the same measured samples
    # throughout optimization so that a pulse cannot reduce residual by moving the gate.
    initial = arrivals(seed)
    half_window = int(np.ceil(16 * seed[5]))
    offsets = np.arange(-half_window, half_window + 1)
    rows = [np.unique((np.rint(t).astype(int)[:, None] + offsets).clip(0, len(y)-1)) for t in initial]
    times = np.zeros((len(rows), max(map(len, rows))), dtype=int)
    keep = np.zeros_like(times, dtype=bool)
    for j, row in enumerate(rows):
        times[j, :len(row)] = row
        keep[j, :len(row)] = True
    obs = y[times, np.arange(len(xy))[:, None]] * keep

    def residual(params):
        centers = arrivals(params)
        atoms = _pulse(times[:, :, None] - centers[:, None, :], params[5])
        res, amps = _project(atoms * keep[:, :, None], obs)
        return np.r_[res.ravel(), np.sqrt(1e-7) * amps.ravel()]

    lo = [-32.] * 4 + [_SPEED_BOUNDS[0], seed[5] * 0.5]
    hi = [32.] * 4 + [_SPEED_BOUNDS[1], seed[5] * 2.0]
    return least_squares(residual, seed, bounds=(lo, hi), x_scale=[1, 1, 1, 1, 30, 1],
                         ftol=1e-8, max_nfev=45, tr_solver='lsmr').x


def _restart(y, xy, pin_distance, factor, result):
    """Reopen the source search only if the joint fit leaves signal above noise."""
    time = np.arange(len(y))[None, :, None]
    noise = np.median(np.abs(np.diff(y, axis=0))) / (0.67449 * np.sqrt(2.0))

    def direct(points):
        return np.linalg.norm(xy[:, None, :] - points[None, :, :], axis=2)

    def residual(params):
        distances = np.column_stack((direct(params[:4].reshape(2, 2)), pin_distance))
        centers = distances * (factor / params[4])
        return _project(_pulse(time - centers[:, None, :], params[5]), y.T)[0]

    best = result
    best_rms = np.sqrt(np.mean(residual(best) ** 2))
    noise_limit = 1.12 * noise
    if best_rms <= noise_limit:
        return best

    axis = np.arange(-32.0, 32.01, 0.5)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = grid[np.sum(grid * grid, axis=1) <= 32.0 ** 2]
    scale = factor / result[4]
    grid_t = direct(grid).T * scale
    width = result[5]

    def overlap(delta):
        v = (delta / (2.0 * width)) ** 2
        return (1.0 - 2.0 * v) * np.exp(-v)

    for fixed in result[:4].reshape(2, 2):
        centers = np.column_stack((direct(fixed[None, :]), pin_distance)) * scale
        r = _project(_pulse(time - centers[:, None, :], width), y.T)[0]
        coeff, _ = _matched(r.T, width)
        gram = overlap(centers[:, :, None] - centers[:, None, :])
        inverse = np.linalg.inv(gram + 1e-6 * np.eye(2))
        cross = overlap(grid_t[..., None] - centers)
        norm = 1.0 - np.einsum('gsi,sij,gsj->gs', cross, inverse, cross)
        score = np.sum(_sample(coeff, grid_t) ** 2 / np.maximum(norm, 0.10), axis=1)
        for _ in range(4):
            point = grid[int(np.argmax(score))]
            score[np.sum((grid - point) ** 2, axis=1) < 0.8 ** 2] = -np.inf
            candidate = _fit(y, xy, pin_distance, factor, np.r_[fixed, point, result[4:]])
            rms = np.sqrt(np.mean(residual(candidate) ** 2))
            if rms < best_rms:
                best, best_rms = candidate, rms
                if rms <= noise_limit:
                    return best
    return best


def reconstruct_two_absorbers_mm(scan: dict) -> dict:
    """Return the common sound speed and two absorber coordinates sorted by x."""
    y = np.asarray(scan['signals'], dtype=float)
    xy = np.asarray(scan['sensor_xy_mm'], dtype=float)
    pin = np.asarray(scan['calib_pin_xy_mm'], dtype=float)
    factor = float(scan['fs_hz']) / 1000.0  # Converts mm/(m/s) to samples.
    pin_distance = np.linalg.norm(xy - pin, axis=1)
    seed = _seed(y, xy, pin_distance, factor, 4.4 * factor / 20000.0)
    result = _fit(y, xy, pin_distance, factor, seed)
    result = _restart(y, xy, pin_distance, factor, result)
    points = result[:4].reshape(2, 2)
    return {'sound_speed_m_per_s': float(result[4]),
            'absorbers_xy_mm': points[np.argsort(points[:, 0])].tolist()}
