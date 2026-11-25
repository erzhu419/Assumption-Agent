"""Straight-ray inversion for a Gaussian plume and receiver clock offsets."""

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares
from scipy.special import ndtr


def _project_clocks(clock):
    """Project onto the zero-sum clock box (units: milliseconds)."""
    if np.max(np.abs(clock)) <= 4:
        return clock
    lo, hi = clock.min() - 4, clock.max() + 4
    for _ in range(50):
        shift = (lo + hi) / 2
        if np.clip(clock - shift, -4, 4).sum() > 0:
            lo = shift
        else:
            hi = shift
    return np.clip(clock - (lo + hi) / 2, -4, 4)


def _invert(survey):
    width, height = survey['W_m'], survey['H_m']
    radius = survey.get('R_plume_m', 5.5)
    sources = np.asarray(survey['z_src_m'])
    receivers = np.asarray(survey['z_rcv_m'])
    data = np.asarray(survey['T_obs_s'], dtype=float).ravel() * 1000
    ns, nr = len(sources), len(receivers)
    source_z = np.repeat(sources, nr)
    dz = np.tile(receivers, ns) - source_z
    length = np.hypot(width, dz)
    c, s = width / length, dz / length
    clock_basis = helmert(nr, full=False).T
    linear = np.column_stack((length / width, -length / width * s**2,
                              np.tile(clock_basis, (ns, 1))))
    nlinear = linear.shape[1]

    def plume(x, z, derivatives=False):
        rel_z = z - source_z
        along = c*x + s*rel_z
        perp = c*rel_z - s*x
        a, b = -along/radius, (length-along)/radius
        taper = np.exp(-0.5*(perp/radius)**2)
        value = taper * (ndtr(b) - ndtr(a))
        if not derivatives:
            return value
        dalong = taper * (np.exp(-0.5*a*a) - np.exp(-0.5*b*b)) / (radius*np.sqrt(2*np.pi))
        dperp = -perp / radius**2 * value
        return value, dalong*c - dperp*s, dalong*s + dperp*c

    def predict(p):
        return linear @ p[:nlinear] + p[-1]*plume(*p[-3:-1])

    def jacobian(p):
        value, dx, dz = plume(*p[-3:-1], derivatives=True)
        return np.column_stack((linear, p[-1]*dx, p[-1]*dz, value))

    # Remove the linear nuisance subspace before looking for plume-shaped signal.
    # A robust background fit supplies weights so cycle skips cannot dominate the scan.
    initial = np.linalg.lstsq(linear, data, rcond=None)[0]
    bg = least_squares(lambda p: linear @ p - data, initial, jac=lambda p: linear.copy(),
                       loss='soft_l1', f_scale=.35).x
    weights = (1 + ((linear @ bg - data) / 1.2)**4)**(-.25)
    q, _ = np.linalg.qr(linear * weights[:, None])
    xy = np.array([(x, z) for x in np.linspace(0, width, int(np.ceil(width/2.5))+1)
                   for z in np.linspace(0, height, int(np.ceil(height/2.5))+1)])
    templates = plume(xy[:, 0, None], xy[:, 1, None]) * weights
    templates -= (templates @ q) @ q.T
    norms = np.maximum(np.linalg.norm(templates, axis=1), 1e-10)
    scores = (templates @ (weights * (data - linear @ bg))) / norms
    lo = np.r_[np.full(nlinear, -np.inf), 0, 0, 0]
    hi = np.r_[np.full(nlinear, np.inf), width, height, np.inf]
    starts = []
    best = None
    for idx in np.argsort(scores)[::-1]:
        center = xy[idx]
        if any(np.linalg.norm(center - prev) < radius for prev in starts):
            continue
        starts.append(center)
        amp = max(scores[idx] / norms[idx], .05)
        init = np.r_[bg, center, amp]
        fit = least_squares(lambda p: predict(p) - data, init, jac=jacobian,
                            bounds=(lo, hi), loss='soft_l1', f_scale=.20, max_nfev=65)
        if best is None or fit.cost < best.cost:
            best = fit
        if len(starts) == 8:
            break

    # Refit the picked inliers without the residual influence of cycle skips.
    p = best.x
    inlier = np.abs(predict(p) - data) < 3.0
    p = least_squares(lambda p: (predict(p) - data)[inlier], p,
                      jac=lambda p: jacobian(p)[inlier], bounds=(lo, hi)).x
    clocks = _project_clocks(clock_basis @ p[2:nlinear])
    return p[-3:-1], clocks / 1000, p, data - predict(p), inlier


def invert_plume_and_clocks(survey: dict) -> dict:
    """Recover plume center (meters) and zero-sum receiver offsets (seconds)."""
    center, clocks, *_ = _invert(survey)
    return {'plume_xz_m': center.tolist(), 'rcv_clock_bias_s': clocks.tolist()}
