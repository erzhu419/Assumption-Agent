"""Straight-ray Gaussian plume tomography with receiver clock calibration."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


def _bounded_clocks(clocks):
    """Project milliseconds onto the zero-sum clock box, if necessary."""
    if np.max(np.abs(clocks)) <= 4.0:
        return clocks
    lo, hi = np.min(clocks) - 4.0, np.max(clocks) + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clocks - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clocks - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate the plume center (m) and receiver timing corrections (s)."""
    w = float(survey['W_m'])
    h = float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    sources = np.asarray(survey['z_src_m'], dtype=float)
    receivers = np.asarray(survey['z_rcv_m'], dtype=float)
    times = 1000 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    ns, nr = len(sources), len(receivers)
    zs = np.repeat(sources, nr)
    dz = np.tile(receivers, ns) - zs
    length = np.hypot(w, dz)
    ux, uz = w / length, dz / length
    # Linear coefficients are s0 and s0*epsilon, in ms/m, followed by
    # independent clock offsets; the last receiver closes the zero-sum gauge.
    basis = np.vstack((np.eye(nr - 1), -np.ones(nr - 1)))
    linear = np.column_stack((length, -dz * uz, np.tile(basis, (ns, 1))))

    def integral(x, z):
        """Analytic line integral between the two wells, not an infinite ray."""
        depth = z - zs
        along = (ux * x + uz * depth) / radius
        cross = (ux * depth - uz * x) / radius
        return (np.sqrt(2 * np.pi) * radius * np.exp(-0.5 * cross**2)
                * (ndtr(length / radius - along) - ndtr(-along)))

    def dot_columns(a, b):
        return np.einsum('ij,i->j', a, b)

    def matvec(a, b):
        return np.einsum('ij,j->i', a, b)

    # Background-only Huber fit gives a first mask without allowing a skipped
    # pick to drag its entire receiver column away from the other picks.
    weights = np.ones_like(times)
    for _ in range(12):
        coef = np.linalg.lstsq(linear * weights[:, None], times * weights, rcond=None)[0]
        residual = times - matvec(linear, coef)
        weights = np.sqrt(0.25 / np.maximum(0.25, np.abs(residual)))
    mask = np.abs(residual) < 3.0

    def fit_center(mask, seeds):
        a = linear[mask]
        y = times[mask]
        q, _ = np.linalg.qr(a, mode='reduced')
        y_perp = y - matvec(q, dot_columns(q, y))

        def profile(center):
            g = integral(*center)[mask]
            g -= matvec(q, dot_columns(q, g))
            norm2 = g @ g
            if norm2 == 0:
                return 0.0
            return -max(0.0, g @ y_perp)**2 / norm2

        if seeds is None:
            # A whole-domain coarse search avoids a local plume-center minimum.
            step = radius / 2
            xs = np.linspace(0, w, int(np.ceil(w / step)) + 1)
            zs_grid = np.linspace(0, h, int(np.ceil(h / step)) + 1)
            grid = [(x, z) for x in xs for z in zs_grid]
            ranked = sorted(grid, key=profile)
            seeds = []
            for point in ranked:
                if all(np.hypot(point[0] - x, point[1] - z) >= radius for x, z in seeds):
                    seeds.append(point)
                    if len(seeds) == 5:
                        break

        fits = [minimize(profile, seed, method='L-BFGS-B',
                         bounds=((0, w), (0, h)), options={'ftol': 1e-12})
                for seed in seeds]
        best = min(fits, key=lambda fit: fit.fun)
        g_all = integral(*best.x)
        g = g_all[mask]
        g_perp = g - matvec(q, dot_columns(q, g))
        norm2 = g_perp @ g_perp
        amplitude = max(0.0, (g_perp @ y_perp) / norm2) if norm2 else 0.0
        coeff = np.linalg.lstsq(a, y - amplitude * g, rcond=None)[0]
        return best.x, coeff, times - matvec(linear, coeff) - amplitude * g_all

    center, coef, residual = fit_center(mask, None)
    for _ in range(3):
        updated = np.abs(residual) < 2.5
        if np.array_equal(updated, mask):
            break
        mask = updated
        center, coef, residual = fit_center(mask, [center])

    clocks = _bounded_clocks(basis @ coef[2:]) / 1000
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': clocks.tolist()}
