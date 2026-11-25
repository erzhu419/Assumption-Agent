"""Robust straight-ray inversion of cross-well arrival times."""

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.special import ndtr


def _clock_projection(clock):
    """Project millisecond offsets onto the zero-sum, ±4 ms clock set."""
    clock = clock - clock.mean()
    if np.max(np.abs(clock)) <= 4.0:
        return clock
    lo, hi = clock.min() - 4.0, clock.max() + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Fit plume location and receiver clocks; all input/output times are seconds."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    obs = 1000.0 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    zs = np.repeat(src, rcv.size)
    dz = np.tile(rcv, src.size) - zs
    length = np.hypot(width, dz)
    cos, sin = width / length, dz / length

    # Linear background coefficients are s0 and -s0*eps. Use a
    # contrast basis so the clock offsets cannot absorb a common delay.
    clock_basis = np.eye(rcv.size)[:, :-1] - np.eye(rcv.size)[:, -1:]
    design = np.column_stack((length, length * sin**2,
                              np.tile(clock_basis, (src.size, 1))))

    def template(x, z):
        depth = z - zs
        along = cos * x + sin * depth
        across = cos * depth - sin * x
        # Integrate the Gaussian between the actual source and receiver.
        return (np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * (across / radius)**2)
                * (ndtr((length - along) / radius) - ndtr(-along / radius)))

    # A robust background-only fit gives a first mask. Ordinary plume
    # delays in these surveys are much smaller than a cycle skip.
    init = np.linalg.lstsq(design, obs, rcond=None)[0]
    base = least_squares(lambda v: design @ v - obs, init,
                         jac=lambda v: design.copy(), loss='huber', f_scale=0.18)
    good = np.abs(base.fun) < 2.8
    bounds = ([0.0, 0.0], [width, height])
    gx = np.linspace(0, width, int(np.ceil(2 * width / radius)) + 1)
    gz = np.linspace(0, height, int(np.ceil(2 * height / radius)) + 1)
    grid = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)

    for _ in range(4):
        # Variable projection: eliminate background, clocks and nonnegative
        # plume strength, leaving just the two coordinates to search.
        q, r = qr(design[good], mode='economic')
        qt = q.T.copy()
        target = obs[good]
        qty = qt @ target
        y = target - q @ qty

        def project(p):
            return p - q @ (qt @ p)

        def score(p):
            pp = project(p)
            return max(0.0, float(pp @ y)) / np.sqrt(max(float(pp @ pp), 1e-24))

        def profiled(xz):
            pp = project(template(*xz)[good])
            amp = max(0.0, float(pp @ y)) / max(float(pp @ pp), 1e-24)
            return amp * pp - y

        start = max(grid, key=lambda xy: score(template(*xy)[good]))
        xy = least_squares(profiled, start, bounds=bounds, ftol=1e-10,
                           xtol=1e-9, gtol=1e-9).x
        p = template(*xy)
        pp = project(p[good])
        amp = max(0.0, float(pp @ y)) / max(float(pp @ pp), 1e-24)
        coef = solve_triangular(r, qty - amp * (qt @ p[good]))
        refined = np.abs(obs - design @ coef - amp * p) < 2.8
        if np.array_equal(refined, good):
            break
        good = refined

    clock = _clock_projection(clock_basis @ coef[2:])
    return {'plume_xz_m': xy.tolist(), 'rcv_clock_bias_s': (clock / 1000.0).tolist()}
