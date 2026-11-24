"""Robust strapdown magnetometer calibration with unknown calibration yaw."""
import numpy as np
from scipy.optimize import least_squares


def _level(rp):
    """Rows of the body-to-level rotation R_y(pitch) R_x(roll)."""
    p, t = rp.T
    sp, cp, st, ct = np.sin(p), np.cos(p), np.sin(t), np.cos(t)
    z = np.zeros_like(p)
    return np.stack((ct, st*sp, st*cp, z, cp, -sp,
                     -st, ct*sp, ct*cp), axis=1).reshape(-1, 3, 3)


def _symmetric(q, trace):
    a, b, c, d, e = q
    return np.array([[a, c, d], [c, b, e], [d, e, trace-a-b]])


def _pack(A):
    return A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2]


def _exp(q):
    w, u = np.linalg.eigh(_symmetric(q, 0.))
    return (u * np.exp(w)) @ u.T


def _vertical_seed(x, v):
    # A is symmetric and has trace 3 during this linear fit. The known
    # down direction gives v @ (A @ x + c) = D at each attitude.
    a, b, c = x.T
    i, j, k = v.T
    X = np.column_stack((i*a-k*c, j*b-k*c, i*b+j*a,
                         i*c+k*a, j*c+k*b, v, -np.ones(len(x))))
    y = -3*k*c
    rng = np.random.default_rng(176)
    best, score = None, np.inf
    for ids in [slice(None)] + [rng.choice(len(x), min(18, len(x)), replace=False) for _ in range(80)]:
        q = np.linalg.lstsq(X[ids], y[ids], rcond=None)[0]
        A = _symmetric(q[:5], 3.)
        if np.linalg.eigvalsh(A)[0] <= 0.05:
            continue
        s = np.median(np.abs(X @ q - y) / np.linalg.norm(v @ A, axis=1))
        if s < score:
            best, score = q, s
    if best is None:
        return np.eye(3), np.zeros(3)
    q = best
    # Use the heavy-tailed MAD/IRLS principle with the appropriate
    # attitude-dependent projection of raw magnetometer noise.
    for _ in range(7):
        g = np.linalg.norm(v @ _symmetric(q[:5], 3.), axis=1)
        r = (X @ q - y) / g
        s = max(1.4826*np.median(np.abs(r-np.median(r))), 1e-7)
        w = 1 / (g*np.hypot(1., r/(2.5*s)))
        q = np.linalg.lstsq(X*w[:, None], y*w, rcond=None)[0]
        if np.linalg.eigvalsh(_symmetric(q[:5], 3.))[0] <= 0.05:
            q = best
            break
    A = _symmetric(q[:5], 3.)
    return A/np.cbrt(np.linalg.det(A)), -np.linalg.solve(A, q[5:8])


def _fit(x, L, joint=True):
    A, b = _vertical_seed(x, L[:, 2])
    if not joint:
        return A, b
    m = np.einsum('nij,nj->ni', L, (x-b) @ A)
    w, u = np.linalg.eigh(A)
    q = np.r_[_pack((u*np.log(w)) @ u.T), b,
              np.median(np.linalg.norm(m[:, :2], axis=1)), np.median(m[:, 2])]

    def residual(q, ids):
        A = _exp(q[:5]); l = L[ids]
        m = np.einsum('nij,nj->ni', l, (x[ids]-q[5:8]) @ A)
        h = np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-12)
        g = ((m[:, :2]/h[:, None])[:, :, None] * l[:, :2]).sum(axis=1) @ A
        v = l[:, 2] @ A
        # Whiten the two invariants under isotropic raw-frame noise.
        vn = np.linalg.norm(v, axis=1)
        vu = v/vn[:, None]
        cross = np.sum(g*vu, axis=1)
        zd = (m[:, 2]-q[9])/vn
        rh = (h-q[8]-cross*zd)/np.linalg.norm(g-cross[:, None]*vu, axis=1)
        return np.column_stack((rh, zd))

    ids = np.arange(len(x))
    for _ in range(4):
        r = residual(q, ids)
        s = max(1.4826*np.median(np.abs(r-np.median(r, axis=0))), 1e-7)
        wt = 1/np.hypot(1., np.linalg.norm(r, axis=1)/(2.5*s))
        q = least_squares(lambda p: (residual(p, ids)*wt[:, None]).ravel(), q,
                          max_nfev=40).x
    r = residual(q, ids)
    s = max(1.4826*np.median(np.abs(r-np.median(r, axis=0))), 1e-7)
    inlier = ids[np.linalg.norm(r, axis=1) < 4.25*s]
    if len(inlier) >= 10:
        q = least_squares(lambda p: residual(p, inlier).ravel(), q, max_nfev=40).x
    return _exp(q[:5]), q[5:8]


def calibrate_and_estimate_heading(session: dict) -> dict:
    """Return S_soft, b_hard (µT), and the test headings in degrees."""
    x = np.asarray(session['calib_m_raw'], dtype=float)
    L = _level(np.asarray(session['calib_roll_pitch_rad'], dtype=float))
    origin = np.median(x, axis=0)
    scale = np.median(np.linalg.norm(x-origin, axis=1))
    A, b = _fit((x-origin)/scale, L)
    bias = origin + scale*b
    T = _level(np.asarray(session['test_roll_pitch_rad'], dtype=float))
    m = np.einsum('nij,nj->ni', T, (np.asarray(session['test_m_raw'])-bias) @ A)
    heading = np.degrees(np.arctan2(-m[:, 1], m[:, 0]))
    heading[heading <= -180.] = 180.
    S = np.linalg.inv(A)
    return {'S_soft': (S+S.T)/2, 'b_hard': bias, 'heading_deg': heading}
