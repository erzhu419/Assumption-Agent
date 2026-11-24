"""Robust, determinant-one magnetometer calibration from roll and pitch."""
import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.stack((np.stack((cp, sp * sr, sp * cr), axis=1),
                     np.stack((np.zeros_like(cr), cr, -sr), axis=1),
                     np.stack((-sp, cp * sr, cp * cr), axis=1)), axis=1)


def _symmetric(p, trace=0.):
    a, b, xy, xz, yz = p
    return np.array([[a, xy, xz], [xy, b, yz], [xz, yz, trace - a - b]])


def _correction(p):
    eigenvalues, axes = np.linalg.eigh(_symmetric(p[:5]))
    return (axes * np.exp(eigenvalues)) @ axes.T


def _seed(x, axes):
    # The vertical equation is linear in W, W*b and field-down.
    # Fix trace(W)=3 to solve it, then convert to the determinant gauge.
    v = axes[:, 2]
    a, b, c = v.T
    i, j, k = x.T
    mat = np.column_stack((a*i-c*k, b*j-c*k, a*j+b*i,
                           a*k+c*i, b*k+c*j, -v, -np.ones(len(x))))
    rhs = -3*c*k
    z = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    for _ in range(18):
        error = mat @ z - rhs
        scale = max(1.4826 * np.median(abs(error - np.median(error))), 1e-6)
        weight = 1 / np.hypot(1, error / (2.5 * scale))
        z1 = np.linalg.lstsq(mat * weight[:, None], rhs * weight, rcond=None)[0]
        if np.linalg.norm(z1 - z) < 1e-8:
            break
        z = z1
    w = _symmetric(z[:5], trace=3.)
    val, vec = np.linalg.eigh(w)
    if val[0] > .08:
        bias = np.linalg.solve(w, z[5:8])
        logval = np.log(val)
        logw = (vec * (logval - logval.mean())) @ vec.T
    else:
        bias = np.zeros(3)
        logw = np.zeros((3, 3))
    p = np.r_[logw[0, 0], logw[1, 1], logw[0, 1], logw[0, 2], logw[1, 2], bias]
    level = np.einsum('nij,nj->ni', axes, (x - bias) @ _correction(p))
    return np.r_[p, np.median(np.linalg.norm(level[:, :2], axis=1)), np.median(level[:, 2])]


def _residual(p, x, axes):
    w = _correction(p)
    level = np.einsum('nij,nj->ni', axes, (x - p[5:8]) @ w)
    h = np.linalg.norm(level[:, :2], axis=1)
    # Both residuals come from the same raw 3-vector. Whiten their local
    # covariance so the loss measures distance in raw sensor units.
    hdir = level[:, :2] / np.maximum(h[:, None], 1e-12)
    radial = np.einsum('ni,nij->nj', hdir, axes[:, :2]) @ w
    vertical = axes[:, 2] @ w
    vz = np.linalg.norm(vertical, axis=1)
    unit_z = vertical / vz[:, None]
    cross = np.einsum('ni,ni->n', radial, unit_z)
    orth = np.linalg.norm(radial - cross[:, None] * unit_z, axis=1)
    dz = (level[:, 2] - p[9]) / vz
    dh = (h - p[8] - cross * dz) / np.maximum(orth, 1e-12)
    return np.column_stack((dz, dh))


def _fit(x, axes):
    p = _seed(x, axes)
    for _ in range(12):
        residual = _residual(p, x, axes)
        distance = np.linalg.norm(residual, axis=1)
        # The median of a two-dimensional Gaussian radius is sigma*sqrt(2 ln 2).
        sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
        weight = 1 / np.sqrt(1 + (distance / (2.5 * sigma))**2)
        fit = least_squares(lambda q: (_residual(q, x, axes) * weight[:, None]).ravel(),
                            p, max_nfev=40)
        change = np.linalg.norm(fit.x - p)
        p = fit.x
        if change < 1e-7:
            break

    # Once the fit is close, reject large spikes and restore equal weights
    # to the inliers. Both coordinates are screened together per observation.
    for _ in range(4):
        distance = np.linalg.norm(_residual(p, x, axes), axis=1)
        sigma = max(np.median(distance) / np.sqrt(2 * np.log(2)), 1e-6)
        keep = distance <= 4 * sigma
        q = least_squares(lambda q: _residual(q, x[keep], axes[keep]).ravel(),
                          p, max_nfev=25).x
        change = np.linalg.norm(q - p)
        p = q
        if change < 1e-7:
            break
    return p


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Recover physical soft- and hard-iron parameters and test magnetic yaw."""
    raw = np.asarray(session['calib_m_raw'], dtype=float)
    axes = _level_axes(session['calib_roll_pitch_rad'])
    origin = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw - origin, axis=1))
    p = _fit((raw - origin) / scale, axes)
    correction = _correction(p)
    bias = origin + scale * p[5:8]
    test = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ correction
    level = np.einsum('nij,nj->ni', _level_axes(session['test_roll_pitch_rad']), test)
    heading = np.rad2deg(np.arctan2(-level[:, 1], level[:, 0]))
    heading = np.where(heading <= -180., 180., heading)
    return {'S_soft': np.linalg.inv(correction), 'b_hard': bias, 'heading_deg': heading}
