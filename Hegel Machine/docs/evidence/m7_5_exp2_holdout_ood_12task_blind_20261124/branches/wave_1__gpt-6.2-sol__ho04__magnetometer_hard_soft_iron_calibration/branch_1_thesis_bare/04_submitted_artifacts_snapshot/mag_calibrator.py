"""Robust hard- and soft-iron calibration from known roll and pitch."""

import numpy as np
from scipy.optimize import least_squares


def _level_rotation(roll_pitch):
    """Body to yaw-rotated NED: Ry(pitch) @ Rx(roll)."""
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    sr, cr = np.sin(roll), np.cos(roll)
    st, ct = np.sin(pitch), np.cos(pitch)
    return np.stack(((ct, st * sr, st * cr),
                     (np.zeros_like(sr), cr, -sr),
                     (-st, ct * sr, ct * cr)), axis=-1).transpose(1, 2, 0)


def _vertical_start(raw, down):
    """Fit n.T C x - n.T d = D, where d=C b, with trace(C)=3."""
    x, y, z = raw.T
    u, v, w = down.T
    design = np.column_stack((u*x - w*z, v*y - w*z,
                              u*y + v*x, u*z + w*x, v*z + w*y,
                              -u, -v, -w, -np.ones(len(raw))))
    target = -3*w*z
    rng = np.random.default_rng(0)
    best = np.inf
    for i in range(48 if len(raw) > 16 else 1):
        rows = slice(None) if i == 0 else rng.choice(len(raw), 16, replace=False)
        p = np.linalg.lstsq(design[rows], target[rows], rcond=None)[0]
        score = np.median(np.abs(design @ p - target))
        if score < best:
            best, estimate = score, p
    for _ in range(3):
        error = np.abs(design @ estimate - target)
        keep = error <= max(np.quantile(error, 0.85), 4.5*np.median(error))
        estimate = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    a, b, xy, xz, yz = estimate[:5]
    inverse = np.array([[a, xy, xz], [xy, b, yz], [xz, yz, 3-a-b]])
    if np.linalg.eigvalsh(inverse)[0] > 0:
        center = np.linalg.solve(inverse, estimate[5:8])
        inverse /= np.cbrt(np.linalg.det(inverse))
    else:
        inverse = np.eye(3)
        center = np.linalg.lstsq(np.column_stack((down, np.ones(len(raw)))),
                                 np.sum(down * raw, axis=1), rcond=None)[0][:3]
    return inverse, center


def _inverse_matrix(params):
    a, b, xy, xz, yz = params[:5]
    lower = np.array([[np.exp(a), 0., 0.], [xy, np.exp(b), 0.],
                      [xz, yz, np.exp(-a-b)]])
    return lower @ lower.T


def _fit(raw, rotation):
    # Center and scale raw vectors for conditioning; the correction matrix
    # itself stays dimensionless and has determinant one throughout the fit.
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw - origin, axis=1))
    raw = (raw - origin) / scale
    inverse, center = _vertical_start(raw, rotation[:, 2, :])
    lower = np.linalg.cholesky(inverse)
    level = np.einsum('nij,nj->ni', rotation, (raw - center) @ inverse)
    start = np.r_[np.log(lower[0, 0]), np.log(lower[1, 1]),
                  lower[1, 0], lower[2, 0], lower[2, 1], center,
                  np.median(np.hypot(level[:, 0], level[:, 1])), np.median(level[:, 2])]

    def residual(params, samples, leveling):
        inverse = _inverse_matrix(params)
        transform = leveling @ inverse
        level = np.einsum('nij,nj->ni', transform, samples - params[5:8])
        h = np.hypot(level[:, 0], level[:, 1])
        # Map cylinder constraint errors to first-order raw-space distances.
        # This accounts for distortion of the measurement noise by the inverse.
        hdir = level[:, :2] / np.maximum(h[:, None], 1e-15)
        gh = np.einsum('ni,nij->nj', hdir, transform[:, :2, :])
        gz = transform[:, 2, :]
        gz_norm = np.linalg.norm(gz, axis=1)
        gz_unit = gz / gz_norm[:, None]
        projection = np.sum(gh * gz_unit, axis=1)
        gh_norm = np.linalg.norm(gh - projection[:, None] * gz_unit, axis=1)
        rz = (level[:, 2] - params[9]) / gz_norm
        rh = (h - params[8] - projection * rz) / np.maximum(gh_norm, 1e-15)
        return np.column_stack((rh, rz))

    def robust_residual(params, width):
        r = residual(params, raw, rotation)
        t = np.sum((r / width)**2, axis=1)
        # One Cauchy loss per sample, not one per coordinate.
        weight = np.sqrt(np.log1p(t) / np.maximum(t, 1e-30))
        return (r * np.where(t < 1e-30, 1., weight)[:, None]).ravel()

    width = max(np.median(np.linalg.norm(residual(start, raw, rotation), axis=1)), 1e-8)
    params = least_squares(robust_residual, start, args=(2*width,), max_nfev=150).x
    width = max(np.median(np.linalg.norm(residual(params, raw, rotation), axis=1)), 1e-8)
    params = least_squares(robust_residual, params, args=(2*width,), max_nfev=100).x
    # Use ordinary least squares for the clean core, after removing spikes.
    distance = np.linalg.norm(residual(params, raw, rotation), axis=1)
    keep = distance <= max(3.5*np.median(distance), 1e-8)
    params = least_squares(lambda p: residual(p, raw[keep], rotation[keep]).ravel(),
                           params, max_nfev=100).x
    return _inverse_matrix(params), origin + scale*params[5:8]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return estimated soft-iron matrix, hard-iron bias (uT), and headings (deg)."""
    inverse, bias = _fit(np.asarray(session['calib_m_raw'], dtype=float),
                         _level_rotation(session['calib_roll_pitch_rad']))
    field_body = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ inverse
    field_level = np.einsum('nij,nj->ni', _level_rotation(session['test_roll_pitch_rad']),
                           field_body)
    heading = np.rad2deg(np.arctan2(-field_level[:, 1], field_level[:, 0]))
    return {'S_soft_hat': np.linalg.inv(inverse), 'b_hard_hat': bias,
            'heading_deg': np.where(heading <= -180., 180., heading)}
