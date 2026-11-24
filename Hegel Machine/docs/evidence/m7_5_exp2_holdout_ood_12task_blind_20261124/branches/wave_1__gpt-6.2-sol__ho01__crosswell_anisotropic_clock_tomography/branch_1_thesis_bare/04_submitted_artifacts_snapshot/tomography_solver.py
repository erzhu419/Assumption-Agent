"""Straight-ray inversion for a Gaussian plume and receiver clock offsets."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _bounded_clocks(clocks):
    """Project onto the zero-sum, +/-4 ms clock constraint."""
    if np.max(np.abs(clocks)) <= 4.0:
        return clocks
    lo, hi = np.min(clocks) - 4.0, np.max(clocks) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if np.sum(np.clip(clocks - mid, -4.0, 4.0)) > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clocks - (lo + hi) / 2.0, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate the plume center (m) and receiver clock biases (s)."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    y = 1000.0 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    ns, nr = len(src), len(rcv)
    z0 = np.repeat(src, nr)
    dz = np.tile(rcv, ns) - z0
    length = np.hypot(width, dz)
    u, v = width / length, dz / length
    # The two background terms estimate s0 and -s0*epsilon separately.
    # Clock basis columns all sum to zero across the receiver array.
    clock_basis = np.eye(nr)[:, :-1] - 1.0 / nr
    design = np.column_stack((length, length * v**2, np.tile(clock_basis, (ns, 1))))
    scale = np.sqrt(2.0) * radius
    factor = np.sqrt(np.pi / 2.0) * radius

    def kernel(x, z, rows):
        """Integrate the unit-amplitude Gaussian over the finite ray segment."""
        uz, vz = u[rows], v[rows]
        rel_z = z - z0[rows]
        along = uz * x + vz * rel_z
        across = (uz * rel_z - vz * x) / radius
        return factor * np.exp(-0.5 * across**2) * (
            erf((length[rows] - along) / scale) + erf(along / scale)
        )

    # A robust background-only fit removes clock jumps and identifies gross
    # skips before the spatial search. Revisit this classification afterwards.
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(14):
        e = y - design @ beta
        w = (1.0 + (e / 0.18)**2)**(-0.25)
        beta = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
    good = np.abs(y - design @ beta) < 2.5

    def profile(rows):
        a = design[rows]
        q, r = np.linalg.qr(a, mode='reduced')

        def project(values):
            coords = np.einsum('...i,ij->...j', values, q)
            return values - np.einsum('...j,ij->...i', coords, q)

        target = project(y[rows])

        def fit_plume(x, z):
            k = project(kernel(x, z, rows))
            power = np.sum(k * k, axis=-1)
            correlation = np.maximum(0.0, k @ target if k.ndim == 1 else
                                     np.einsum('ij,j->i', k, target))
            amp = correlation / np.maximum(power, 1e-20)
            return k, amp, correlation * amp

        return fit_plume, target, q, r

    fit, _, _, _ = profile(good)
    gx, gz = np.meshgrid(np.linspace(0.0, width, int(np.ceil(width / 2.0)) + 1),
                         np.linspace(0.0, height, int(np.ceil(height / 2.0)) + 1))
    grid = np.column_stack((gx.ravel(), gz.ravel()))
    gains = []
    for block in np.array_split(grid, max(1, int(np.ceil(len(grid) / 128)))):
        gains.extend(fit(block[:, :1], block[:, 1:],)[2])
    center = grid[np.argmax(gains)]

    for refit in range(4):
        fit, target, q, r = profile(good)

        def residual(p):
            k, amp, _ = fit(*p)
            return amp * k - target

        center = least_squares(residual, center, bounds=([0.0, 0.0], [width, height]),
                               max_nfev=50).x
        _, amp, _ = fit(*center)
        plume = amp * kernel(*center, slice(None))
        beta = np.linalg.solve(r, q.T @ (y[good] - plume[good]))
        new_good = np.abs(y - design @ beta - plume) < 2.5
        if np.array_equal(good, new_good):
            break
        good = new_good

    clocks = _bounded_clocks(clock_basis @ beta[2:])
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': (clocks / 1000.0).tolist()}
