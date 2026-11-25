"""Straight-ray inversion of cross-well picks for a Gaussian plume and receiver clocks."""

import numpy as np
from scipy.linalg import helmert, qr
from scipy.optimize import least_squares
from scipy.special import erf


def _clock_box(clock):
    """Project millisecond biases onto the zero-sum box."""
    if np.max(np.abs(clock)) <= 4.0:
        return clock - clock.mean()
    lo, hi = clock.min() - 4.0, clock.max() + 4.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if np.clip(clock - mid, -4.0, 4.0).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(clock - (lo + hi) / 2, -4.0, 4.0)


def invert_plume_and_clocks(survey: dict) -> dict:
    w, h = float(survey['W_m']), float(survey['H_m'])
    radius = float(survey.get('R_plume_m', 5.5))
    src = np.asarray(survey['z_src_m'], dtype=float)
    rcv = np.asarray(survey['z_rcv_m'], dtype=float)
    times = np.asarray(survey['T_obs_s'], dtype=float)
    i, j = np.nonzero(np.isfinite(times))
    zs = src[i]
    dz = rcv[j] - zs
    length = np.hypot(w, dz)
    ux, uz = w / length, dz / length
    y = 1000 * times[i, j]
    # Orthogonal clock coordinates impose the gauge in every fit.
    clocks = helmert(len(rcv), full=False).T
    design = np.column_stack((length, -dz * dz / length, clocks[j]))
    p = design.shape[1]
    rr = np.sqrt(2.0) * radius
    cap = np.sqrt(np.pi / 2.0) * radius

    def plume(x, z, deriv=False):
        along = ux * x + uz * (z - zs)
        cross = -uz * x + ux * (z - zs)
        a, b = along / rr, (length - along) / rr
        env = np.exp(-0.5 * (cross / radius)**2)
        ray = cap * env * (erf(a) + erf(b))
        if not deriv:
            return ray
        dp = env * (np.exp(-a*a) - np.exp(-b*b))
        dn = -cross * ray / (radius * radius)
        return ray, dp * ux - dn * uz, dp * uz + dn * ux

    background = least_squares(lambda c: design @ c - y,
                               np.linalg.lstsq(design, y, rcond=None)[0],
                               jac=lambda c: design.copy(), loss='soft_l1',
                               f_scale=0.45).x
    # A broad mask is only for initialization. Nonlinear fits reconsider all picks.
    keep0 = np.abs(y - design @ background) < 2.7
    if keep0.sum() <= p + 3:
        keep0 = np.ones(len(y), dtype=bool)
    q, _ = qr(design[keep0], mode='economic')
    target = y[keep0] - q @ (q.T @ y[keep0])
    step = radius / 2
    xx, zz = np.meshgrid(np.linspace(0, w, max(2, int(np.ceil(w/step)) + 1)),
                         np.linspace(0, h, max(2, int(np.ceil(h/step)) + 1)))
    xy = np.column_stack((xx.ravel(), zz.ravel()))
    projected = plume(xy[:, :1], xy[:, 1:])[:, keep0]
    projected -= (projected @ q) @ q.T
    energy = np.einsum('ij,ij->i', projected, projected)
    cov = np.maximum(projected @ target, 0.0)
    score = cov * cov / np.maximum(energy, 1e-20)

    lower = np.r_[np.full(p, -np.inf), 0., 0., 0.]
    upper = np.r_[np.full(p, np.inf), np.inf, w, h]

    def residual(v):
        c, amp, x, z = v[:p], *v[p:]
        return design @ c + amp * plume(x, z) - y

    def jacobian(v):
        amp, x, z = v[p:]
        ray, dx, dzp = plume(x, z, True)
        return np.column_stack((design, ray, amp * dx, amp * dzp))

    def fit(v, mask, loss, scale):
        return least_squares(lambda c: residual(c)[mask], v,
                             jac=lambda c: jacobian(c)[mask], bounds=(lower, upper),
                             x_scale='jac', loss=loss, f_scale=scale, max_nfev=100)

    starts = []
    best = None
    all_picks = np.ones(len(y), dtype=bool)
    for k in np.argsort(score)[::-1]:
        x, z = xy[k]
        if any(np.hypot(x-a, z-b) < radius for a, b in starts):
            continue
        starts.append((x, z))
        amp = max(cov[k] / max(energy[k], 1e-20), 0.005)
        c = np.linalg.lstsq(design[keep0], (y - amp * plume(x, z))[keep0], rcond=None)[0]
        result = fit(np.r_[c, amp, x, z], all_picks, 'cauchy', 0.25)
        if best is None or result.cost < best.cost:
            best = result
        if len(starts) == 4:
            break

    v = best.x
    # The cycle-skips are separated by several milliseconds from normal picks.
    mask = np.abs(residual(v)) < 2.5
    if mask.sum() > p + 3:
        v = fit(v, mask, 'linear', 1.0).x
    return {'plume_xz_m': v[-2:].tolist(),
            'rcv_clock_bias_s': (_clock_box(clocks @ v[2:p]) / 1000).tolist()}
