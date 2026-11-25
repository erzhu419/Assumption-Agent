"""Robust straight-ray cross-well traveltime inversion."""
import numpy as np
from scipy.linalg import helmert, qr, solve_triangular
from scipy.optimize import least_squares
from scipy.special import erf


def _geometry(survey):
    w, h = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    zs, zr = map(lambda a: np.asarray(a, dtype=float),
                 (survey['z_src_m'], survey['z_rcv_m']))
    z0 = np.broadcast_to(zs[:, None], (zs.size, zr.size)).ravel()
    dz = (zr[None, :] - zs[:, None]).ravel()
    length = np.hypot(w, dz)
    sin, cos = dz / length, w / length
    contrast = helmert(zr.size, full=False).T
    design = np.column_stack((length / w, -length / w * sin**2,
                              np.tile(contrast, (zs.size, 1))))
    y = 1000.0 * np.asarray(survey['T_obs_s'], dtype=float).ravel()
    ok = np.isfinite(y)
    return w, h, radius, z0[ok], length[ok], sin[ok], cos[ok], contrast, design[ok], y[ok]


def _kernel(x, z, radius, z0, length, sin, cos, derivative=False):
    # Coordinates along and perpendicular to each finite source-receiver segment.
    depth = z - z0
    parallel = x * cos + depth * sin
    normal = -x * sin + depth * cos
    gaussian = np.exp(-0.5 * (normal / radius)**2)
    hi, lo = (length - parallel) / radius, -parallel / radius
    factor = np.sqrt(np.pi / 2) * (erf(hi / np.sqrt(2)) - erf(lo / np.sqrt(2)))
    value = gaussian * factor
    if not derivative:
        return value
    dp = gaussian * (np.exp(-0.5 * lo**2) - np.exp(-0.5 * hi**2)) / radius
    dn = -normal * value / radius**2
    return value, dp * cos - dn * sin, dp * sin + dn * cos


def _invert(survey):
    w, h, radius, z0, length, sin, cos, contrast, design, y = _geometry(survey)
    geom = (radius, z0, length, sin, cos)
    linear = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(6):
        weights = (1 + ((design @ linear - y) / 0.6)**2)**(-0.25)
        linear = np.linalg.lstsq(weights[:, None] * design, weights * y, rcond=None)[0]

    # Profile out background and clocks on the coarse grid; do not mistake a
    # large raw Gaussian response for evidence if the clocks can absorb it.
    weights = (1 + ((design @ linear - y) / 1.0)**2)**(-0.5)
    q, r = qr(weights[:, None] * design, mode='economic')
    yw = weights * y
    yp = yw - q @ (q.T @ yw)
    xs = np.linspace(0, w, max(3, int(np.ceil(w / (radius / 2))) + 1))
    zs = np.linspace(0, h, max(3, int(np.ceil(h / (radius / 2))) + 1))
    points = np.stack(np.meshgrid(xs, zs), axis=-1).reshape(-1, 2)
    grid = weights[:, None] * _kernel(points[:, 0][None, :], points[:, 1][None, :], radius,
                              z0[:, None], length[:, None], sin[:, None], cos[:, None])
    qp = q.T @ grid
    gp = grid - q @ qp
    energy = np.maximum(np.sum(gp**2, axis=0), 1e-12)
    dot = np.maximum(yp @ gp, 0)
    scores = dot**2 / energy
    lower = np.r_[0., 0., 0., np.full(design.shape[1], -np.inf)]
    upper = np.r_[w, h, np.full(design.shape[1] + 1, np.inf)]

    def predict(p):
        return p[2] * _kernel(*p[:2], *geom) + design @ p[3:]

    def jacobian(p):
        g, gx, gz = _kernel(*p[:2], *geom, derivative=True)
        return np.column_stack((p[2] * gx, p[2] * gz, g, design))

    def fit(p, mask, loss, scale):
        return least_squares(lambda p: (predict(p) - y)[mask], p,
                             jac=lambda p: jacobian(p)[mask], bounds=(lower, upper),
                             loss=loss, f_scale=scale, x_scale='jac', max_nfev=60)

    attempts = []
    for idx in np.argsort(scores)[::-1]:
        point = points[idx]
        if any(np.linalg.norm(point - tried) < radius for tried, _ in attempts):
            continue
        amplitude = max(dot[idx] / energy[idx], 0.05)
        nuisance = solve_triangular(r, q.T @ yw - amplitude * qp[:, idx])
        result = fit(np.r_[point, amplitude, nuisance], slice(None), 'soft_l1', 0.35)
        attempts.append((point, result))
        if len(attempts) == 5:
            break
    p = min((result for _, result in attempts), key=lambda result: result.cost).x
    inliers = np.abs(predict(p) - y) < 2.75  # cycle skips are approximately 6 ms
    p = fit(p, inliers, 'linear', 1.0).x
    # Reuse skipped picks only if an independently estimated common signed
    # shift predicts their observed residuals at the ordinary pick-noise scale.
    residual = y - predict(p)
    skipped = ~inliers
    n_skip = int(skipped.sum())
    if n_skip >= 5:
        e = residual[inliers]
        sigma = np.linalg.norm(e) / np.sqrt(max(1, e.size - p.size))
        signed_size = np.abs(residual[skipped])
        step = signed_size.mean()
        loo_error = n_skip / (n_skip - 1) * (signed_size - step)
        # Include uncertainty of the inlier-only prediction at skipped rays.
        J = jacobian(p)
        _, singular, right = np.linalg.svd(J[inliers], full_matrices=False)
        basis = right[singular > singular[0]*1.e-8]
        leverage = np.sum(((J[skipped] @ basis.T) / singular[:len(basis)])**2, axis=1)
        scaled_error = loo_error / np.sqrt(1. + leverage)
        if (4.5 <= step <= 7.5 and np.sqrt(np.mean(scaled_error**2)) <= 1.4*sigma
                and np.max(np.abs(scaled_error)) <= 3.5*sigma):
            sign = np.where(skipped, np.sign(residual), 0.)
            p = least_squares(lambda t: predict(t[:-1]) + sign*t[-1] - y,
                              np.r_[p, step],
                              jac=lambda t: np.column_stack((jacobian(t[:-1]), sign)),
                              bounds=(np.r_[lower, 4.5], np.r_[upper, 7.5]),
                              x_scale='jac', max_nfev=40).x[:-1]
    return p, contrast @ p[5:], y - predict(p), inliers


def invert_plume_and_clocks(survey: dict) -> dict:
    """Return plume center in meters and zero-sum receiver biases in seconds."""
    p, clock, _, _ = _invert(survey)
    if np.max(np.abs(clock)) > 4.0:
        # Euclidean projection onto the intersection of the clock box and gauge.
        lo, hi = clock.min() - 4, clock.max() + 4
        for _ in range(55):
            mid = (lo + hi) / 2
            if np.clip(clock - mid, -4, 4).sum() > 0:
                lo = mid
            else:
                hi = mid
        clock = np.clip(clock - (lo + hi) / 2, -4, 4)
    return {'plume_xz_m': p[:2].tolist(), 'rcv_clock_bias_s': (clock / 1000).tolist()}
