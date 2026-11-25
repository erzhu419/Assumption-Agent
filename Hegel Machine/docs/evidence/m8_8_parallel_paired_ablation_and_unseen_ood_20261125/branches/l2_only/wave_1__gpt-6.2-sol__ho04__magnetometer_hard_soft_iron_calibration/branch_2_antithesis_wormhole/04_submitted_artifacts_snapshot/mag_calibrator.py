"""Robust hard- and soft-iron calibration from tilt-tagged magnetometer data."""

import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    sr, cr = np.sin(roll), np.cos(roll)
    st, ct = np.sin(pitch), np.cos(pitch)
    return (np.column_stack((ct, st * sr, st * cr)),
            np.column_stack((np.zeros_like(cr), cr, -sr)),
            np.column_stack((-st, ct * sr, ct * cr)))


def _initial_inverse(x, down):
    # A has trace 3 at this stage; the fitted intercept vector is A @ bias.
    i, j, k = (down * x).T
    dx, dy, dz = down.T
    xx, yy, zz = x.T
    design = np.column_stack((i-k, j-k, dx*yy+dy*xx, dx*zz+dz*xx,
                              dy*zz+dz*yy, -down, -np.ones(len(x))))
    target = -3 * k
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    rng = np.random.default_rng(241)
    best = np.median(np.abs(design @ coef - target))
    for _ in range(64):
        inds = rng.choice(len(x), min(15, len(x)), replace=False)
        trial, _, rank, _ = np.linalg.lstsq(design[inds], target[inds], rcond=None)
        if rank == 9:
            score = np.median(np.abs(design @ trial - target))
            if score < best:
                coef, best = trial, score
    for _ in range(5):
        error = design @ coef - target
        mad = np.median(np.abs(error - np.median(error)))
        weight = 1 / np.hypot(1, error / max(2 * mad, 1e-10))
        root = np.sqrt(weight)
        coef = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    a, b, xy, xz, yz = coef[:5]
    inverse = np.array([[a, xy, xz], [xy, b, yz], [xz, yz, 3-a-b]])
    eig, vec = np.linalg.eigh(inverse)
    eig = np.maximum(eig, 0.05)
    inverse = (vec * eig) @ vec.T
    bias = np.linalg.solve(inverse, coef[5:8])
    inverse /= np.cbrt(np.linalg.det(inverse))
    return inverse, bias


def _inverse(parameters):
    a, b, u, v, w = parameters[:5]
    L = np.array([[np.exp(a), 0., 0.], [u, np.exp(b), 0.],
                  [v, w, np.exp(-a-b)]])
    return L @ L.T


def _fit(x, axes):
    u, v, d = axes
    inverse, bias = _initial_inverse(x, d)
    L = np.linalg.cholesky(inverse)
    y = (x-bias) @ inverse
    h = np.hypot(np.sum(y*u, axis=1), np.sum(y*v, axis=1))
    z = np.sum(y*d, axis=1)
    p = np.r_[np.log(L[0, 0]), np.log(L[1, 1]), L[1, 0], L[2, 0], L[2, 1],
              bias, np.median(h), np.median(z)]

    def residual(p, inds):
        a = _inverse(p)
        y = (x[inds]-p[5:8]) @ a
        uu, vv, dd = u[inds], v[inds], d[inds]
        hx = np.sum(y*uu, axis=1)
        hy = np.sum(y*vv, axis=1)
        h = np.hypot(hx, hy)
        # Whiten the two constraints under isotropic raw sensor noise.
        safe_h = np.maximum(h, 1e-12)
        dh = ((hx/safe_h)[:, None]*uu + (hy/safe_h)[:, None]*vv) @ a
        dz = dd @ a
        dh[h < 1e-12] = (uu @ a)[h < 1e-12]
        l11 = np.linalg.norm(dh, axis=1)
        l21 = np.sum(dh*dz, axis=1)/l11
        l22 = np.linalg.norm(dz-(l21/l11)[:, None]*dh, axis=1)
        rh = (h-p[8])/l11
        rz = (np.sum(y*dd, axis=1)-p[9]-l21*rh)/l22
        return np.column_stack((rh, rz))

    rows = np.arange(len(x))
    e = np.linalg.norm(residual(p, rows), axis=1)
    scale = max(np.median(e)/np.sqrt(2*np.log(2)), 1e-7)
    p = least_squares(lambda q: residual(q, rows).ravel(), p, loss='cauchy',
                      f_scale=2*scale, x_scale='jac', max_nfev=150).x
    for _ in range(2):
        e = np.linalg.norm(residual(p, rows), axis=1)
        sigma = np.median(e)/np.sqrt(2*np.log(2))
        keep = rows[e <= max(4.25*sigma, 1e-7)]
        p = least_squares(lambda q: residual(q, keep).ravel(), p,
                          x_scale='jac', max_nfev=100).x
    return _inverse(p), p[5:8], p[8:10]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Fit the calibration on the calibration sweep, then return magnetic yaw."""
    raw = np.asarray(session['calib_m_raw'], dtype=float)
    center = np.median(raw, axis=0)
    scale = np.median(np.linalg.norm(raw-center, axis=1))
    inverse, bias, field = _fit((raw-center)/scale,
                                _level_axes(session['calib_roll_pitch_rad']))
    bias = center + scale*bias
    y = (np.asarray(session['test_m_raw'], dtype=float)-bias) @ inverse
    u, v, _ = _level_axes(session['test_roll_pitch_rad'])
    headings = np.rad2deg(np.arctan2(-np.sum(y*v, axis=1), np.sum(y*u, axis=1)))
    headings[headings <= -180] = 180
    soft = np.linalg.solve(inverse, np.eye(3))
    return {'S_soft': (soft+soft.T)/2, 'b_hard': bias,
            'B_ned_uT': scale*np.array([field[0], 0., field[1]]),
            'heading_deg': headings}
