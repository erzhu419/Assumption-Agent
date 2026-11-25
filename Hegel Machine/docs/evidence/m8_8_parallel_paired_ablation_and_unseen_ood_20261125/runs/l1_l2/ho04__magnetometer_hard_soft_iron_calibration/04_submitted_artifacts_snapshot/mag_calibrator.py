"""Robust calibration of a three-axis magnetometer from known roll and pitch."""
import numpy as np
from scipy.optimize import least_squares


def _level_axes(roll_pitch):
    roll, pitch = np.asarray(roll_pitch, dtype=float).T
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    z = np.zeros_like(roll)
    return (np.column_stack((cp, sp * sr, sp * cr)),
            np.column_stack((z, cr, -sr)),
            np.column_stack((-sp, cp * sr, cp * cr)))


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _vertical_seed(x, v):
    # A has trace 3 in this homogeneous linear initializer. Its determinant
    # is normalized afterwards; c is A times the offset in centered units.
    a, b, c = x.T
    u, w, t = v.T
    design = np.column_stack((u*a-t*c, w*b-t*c, u*b+w*a, u*c+t*a,
                              w*c+t*b, -v, -np.ones(len(x))))
    y = -3*t*c
    def matrix(p):
        a, b, c, d, e = p[:5]
        return np.array([[a,c,d], [c,b,e], [d,e,3-a-b]])

    def rank_seed(p):
        A = matrix(p)
        if np.linalg.eigvalsh(A)[0] <= 0.05:
            return np.inf
        e = (design @ p - y) / np.linalg.norm(v @ A, axis=1)
        return np.median(np.abs(e))

    p = np.linalg.lstsq(design, y, rcond=None)[0]
    best = rank_seed(p)
    # Small random subsets give clean hypotheses even with coherent spikes
    # that exert high leverage in the full vertical regression.
    rng = np.random.default_rng(0)
    for _ in range(64):
        idx = rng.choice(len(x), min(len(x), 24), replace=False)
        q = np.linalg.lstsq(design[idx], y[idx], rcond=None)[0]
        score = rank_seed(q)
        if score < best:
            p, best = q, score
    A = matrix(p)
    if not np.isfinite(best):
        return np.eye(3), np.zeros(3)
    return A / np.cbrt(np.linalg.det(A)), np.linalg.solve(A, p[5:8])


def _pack(A, bias, H, D):
    L = np.linalg.cholesky(A)
    return np.r_[bias, np.log(L[0,0]), np.log(L[1,1]), L[1,0], L[2,0], L[2,1], H, D]


def _unpack(p):
    a, b, c, d, e = p[3:8]
    L = np.array([[np.exp(a),0,0], [c,np.exp(b),0], [d,e,np.exp(-a-b)]])
    return L @ L.T


def _fit(raw, angles):
    v = _level_axes(angles)[2]
    center = np.median(raw, axis=0)
    unit = np.median(np.linalg.norm(raw - center, axis=1))
    x = (raw - center) / unit
    A, b = _vertical_seed(x, v)
    z = (x - b) @ A
    vert = np.sum(z*v, axis=1)
    hor = np.linalg.norm(z - vert[:, None]*v, axis=1)
    p = _pack(A, b, np.median(hor), np.median(vert))

    def residual(p):
        A = _unpack(p)
        z = (x - p[:3]) @ A
        vert = np.sum(z*v, axis=1)
        hvec = z - vert[:, None]*v
        hor = np.linalg.norm(hvec, axis=1)
        # Push isotropic raw-space noise through the two invariant gradients.
        gv = v @ A
        gh = (hvec / np.maximum(hor, 1e-12)[:, None]) @ A
        nv = np.linalg.norm(gv, axis=1)
        parallel = np.sum(gh * gv, axis=1) / nv
        nh = np.linalg.norm(gh - (parallel/nv)[:, None]*gv, axis=1)
        rv = (vert - p[9]) / nv
        rh = (hor - p[8] - parallel*rv) / np.maximum(nh, 1e-12)
        return np.column_stack((rv, rh))

    r = residual(p)
    scale = max(_mad(r.ravel()), 1e-7)
    weight = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / (2.5*scale)**2)
    for _ in range(7):
        p = least_squares(lambda q: (weight[:, None]*residual(q)).ravel(), p,
                          x_scale='jac', max_nfev=45).x
        r = residual(p)
        scale = max(_mad(r.ravel()), 1e-7)
        weight = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / (2.5*scale)**2)
    for _ in range(2):
        r = residual(p)
        scale = max(_mad(r.ravel()), 1e-7)
        keep = np.linalg.norm(r, axis=1) < 4.3*scale
        p = least_squares(lambda q: residual(q)[keep].ravel(), p,
                          x_scale='jac', max_nfev=45).x
    return _unpack(p), center + unit*p[:3]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return hard/soft-iron calibration and tilt-compensated magnetic yaw."""
    A, bias = _fit(np.asarray(session['calib_m_raw'], dtype=float),
                   session['calib_roll_pitch_rad'])
    corrected = (np.asarray(session['test_m_raw'], dtype=float) - bias) @ A
    x, y, _ = _level_axes(session['test_roll_pitch_rad'])
    yaw = np.rad2deg(np.arctan2(-np.sum(y*corrected, axis=1),
                               np.sum(x*corrected, axis=1)))
    yaw[yaw <= -180.] = 180.
    return {'S_soft': np.linalg.inv(A), 'b_hard': bias, 'heading_deg': yaw}
