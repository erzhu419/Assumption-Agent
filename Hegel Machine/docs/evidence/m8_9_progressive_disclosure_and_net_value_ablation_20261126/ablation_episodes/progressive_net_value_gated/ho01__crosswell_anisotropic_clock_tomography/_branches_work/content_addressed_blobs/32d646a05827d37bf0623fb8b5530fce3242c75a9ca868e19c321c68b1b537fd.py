"""Robust straight-ray inversion for a Gaussian slowness plume."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def invert_plume_and_clocks(survey: dict) -> dict:
    """Estimate the plume center and zero-sum receiver clocks from pick times."""
    width, height = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    source = np.asarray(survey['z_src_m'], dtype=float)
    receiver = np.asarray(survey['z_rcv_m'], dtype=float)
    times = np.asarray(survey['T_obs_s'], dtype=float)
    si, ri = np.where(np.isfinite(times))
    y = 1000 * times[si, ri]
    z0 = source[si]
    dz = receiver[ri] - z0
    length = np.hypot(width, dz)
    ux, uz = width / length, dz / length
    nr = receiver.size
    # Orthonormal receiver contrasts implement the sum-zero gauge exactly.
    contrasts = np.zeros((nr, nr - 1))
    for j in range(1, nr):
        contrasts[:j, j - 1] = 1 / np.sqrt(j * (j + 1))
        contrasts[j, j - 1] = -j / np.sqrt(j * (j + 1))
    design = np.column_stack((length / width, -length / width * uz**2,
                              contrasts[ri]))
    scale = np.sqrt(2) * radius
    r2 = radius**2
    edge_factor = 1 / (np.sqrt(2 * np.pi) * radius)

    def gaussian(x, z, derivatives=False):
        along = ux * x + uz * (z - z0)
        across = -uz * x + ux * (z - z0)
        envelope = np.exp(-0.5 * across**2 / r2)
        span = 0.5 * (erf(along / scale) + erf((length - along) / scale))
        value = envelope * span
        if not derivatives:
            return value
        da = envelope * edge_factor * (np.exp(-0.5 * along**2 / r2)
                                       - np.exp(-0.5 * (length - along)**2 / r2))
        dp = -value * across / r2
        return value, da * ux - dp * uz, da * uz + dp * ux

    # A robust background-only fit supplies search weights, not final clocks.
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(6):
        root_w = (1 + ((y - design @ beta) / 1.15)**2)**-0.5
        beta = np.linalg.lstsq(root_w[:, None] * design, root_w * y, rcond=None)[0]
    q = np.linalg.qr(root_w[:, None] * design, mode='reduced')[0]
    yw = root_w * (y - design @ beta)
    yw -= q @ (q.T @ yw)
    gx, gz = np.meshgrid(np.linspace(0, width, max(2, int(np.ceil(width / (radius * .55))) + 1)),
                         np.linspace(0, height, max(2, int(np.ceil(height / (radius * .55))) + 1)))
    grid = np.column_stack((gx.ravel(), gz.ravel()))
    for_start = np.empty(len(grid))
    powers = np.empty(len(grid))
    for k, (x, z) in enumerate(grid):
        t = root_w * gaussian(x, z)
        t -= q @ (q.T @ t)
        powers[k] = max(float(t @ t), 1e-14)
        for_start[k] = max(float(t @ yw), 0.)
    scores = for_start**2 / powers
    starts = []
    for k in np.argsort(scores)[::-1]:
        if all(np.linalg.norm(grid[k] - grid[j]) >= radius * 1.35 for j in starts):
            starts.append(k)
            if len(starts) == 8:
                break

    nlin = design.shape[1]
    lower = np.r_[np.full(nlin, -np.inf), 0., 0., 0.]
    upper = np.r_[np.full(nlin, np.inf), np.inf, width, height]

    def predict(p):
        return design @ p[:nlin] + p[-3] * gaussian(p[-2], p[-1])

    def jacobian(p):
        t, tx, tz = gaussian(p[-2], p[-1], True)
        return np.column_stack((design, t, p[-3] * tx, p[-3] * tz))

    def fit(start, target, mask, loss, f_scale, steps):
        return least_squares(lambda p: (predict(p) - target)[mask], start,
                             jac=lambda p: jacobian(p)[mask], bounds=(lower, upper),
                             loss=loss, f_scale=f_scale, x_scale='jac', max_nfev=steps)

    all_picks = np.ones(y.size, dtype=bool)
    best = None
    for k in starts:
        x, z = grid[k]
        a = max(.05, for_start[k] / powers[k])
        b = np.linalg.lstsq(root_w[:, None] * design,
                            root_w * (y - a * gaussian(x, z)), rcond=None)[0]
        result = fit(np.r_[b, a, x, z], y, all_picks, 'cauchy', .6, 60)
        if best is None or result.cost < best.cost:
            best = result
    p = best.x
    # Remove cycle skips before the final non-robust fit; keep the original
    # picks otherwise, without assuming every gross error is exactly 6 ms.
    for _ in range(3):
        mask = np.abs(y - predict(p)) < 3.
        if mask.sum() < nlin + 3:
            break
        p = fit(p, y, mask, 'linear', 1., 40).x

    clocks = contrasts @ p[2:nlin]
    if np.max(np.abs(clocks)) > 4:
        # Project only if the estimated clocks violate their physical bound.
        lo, hi = clocks.min() - 4, clocks.max() + 4
        for _ in range(45):
            mid = (lo + hi) / 2
            if np.clip(clocks - mid, -4, 4).sum() > 0:
                lo = mid
            else:
                hi = mid
        clocks = np.clip(clocks - (lo + hi) / 2, -4, 4)
    return {'plume_xz_m': p[-2:].tolist(), 'rcv_clock_bias_s': (clocks / 1000).tolist()}
