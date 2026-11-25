"""Recover a passive Prony model and forecast stress without load-cell drift."""

import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter


_TAU_MIN, _TAU_MAX = 0.02, 20.0


def _design(t, strain, tau):
    """Integrate the exponential memory exactly for piecewise-linear strain."""
    A = np.zeros((len(t), len(tau) + 1))
    A[:, 0] = strain
    dt = np.diff(t)
    if not dt.size or not len(tau):
        return A
    de = np.diff(strain)
    if np.allclose(dt, dt[0], rtol=1e-11, atol=0):
        z = dt[0] / tau
        for j, (decay, gain) in enumerate(zip(np.exp(-z), -np.expm1(-z) / z), 1):
            A[1:, j] = lfilter([gain], [1.0, -decay], de)
    else:
        z = dt[:, None] / tau
        decay = np.exp(-z)
        drive = de[:, None] * (-np.expm1(-z) / z)
        for i in range(len(dt)):
            A[i + 1, 1:] = decay[i] * A[i, 1:] + drive[i]
    return A


def _project(Q, a):
    if a.ndim == 1:
        return a - Q @ (Q.T @ a)
    qa = np.einsum('ik,ij->kj', Q, a)
    return a - np.einsum('ik,kj->ij', Q, qa)


def _positive_fit(B, y):
    scale = np.linalg.norm(B, axis=0)
    active = scale > (1e-13 * scale.max())
    c = np.zeros(B.shape[1])
    if np.any(active):
        c[active] = nnls(B[:, active] / scale[active], y, maxiter=3000)[0] / scale[active]
    return c


def _robust_screen(A, drift, y):
    """Cauchy IRLS estimates the trend, then rejects isolated gross errors."""
    w = np.ones(len(y))
    floor = max(np.ptp(y), np.max(np.abs(y))) * 1e-12 + np.finfo(float).tiny
    for _ in range(12):
        Dw = drift * w[:, None]
        Q = qr(Dw, mode='economic')[0]
        c = _positive_fit(_project(Q, A * w[:, None]), _project(Q, y * w))
        physical_residual = y - np.einsum('ij,j->i', A, c)
        b = np.linalg.lstsq(Dw, physical_residual * w, rcond=None)[0]
        r = physical_residual - drift @ b
        centered = r - np.median(r)
        s = max(1.4826 * np.median(np.abs(centered)), floor)
        w = 1 / np.hypot(1, centered / (2.4 * s))
    keep = np.abs(centered) <= 6 * s
    if np.count_nonzero(keep) < drift.shape[1] + 1:
        keep = np.ones(len(y), dtype=bool)
    return keep, s


def _fit(t, strain, y):
    span = np.ptp(t)
    drift = np.column_stack((np.ones(len(t)), (t - t.mean()) / span))
    tau = np.geomspace(_TAU_MIN, _TAU_MAX, 96)
    A = _design(t, strain, tau)
    keep, s = _robust_screen(A, drift, y)
    Q = qr(drift[keep], mode='economic')[0]
    obs = _project(Q, y[keep])
    B = _project(Q, A[keep])
    grid_fit = _positive_fit(B, obs)
    inds = np.flatnonzero(grid_fit[1:] > 1e-7 * grid_fit.max())
    if not len(inds):
        return tau, grid_fit

    # Adjacent grid points often share a single off-grid mode.
    groups = np.split(inds, np.flatnonzero(np.diff(inds) > 2) + 1)
    z0 = np.array([np.average(np.log(tau[g]), weights=grid_fit[g + 1]) for g in groups])
    bounds = np.log(_TAU_MIN), np.log(_TAU_MAX)
    z0 = np.clip(z0, *bounds)

    def evaluate(z):
        B = _project(Q, _design(t, strain, np.exp(z))[keep])
        c = _positive_fit(B, obs)
        return c, (obs - B @ c) / s

    def refine(z, strength):
        def residual(z):
            _, r = evaluate(z)
            return np.r_[r, strength * np.diff(np.sort(z), n=2)] if strength else r

        solved = least_squares(residual, z, bounds=bounds, max_nfev=65)
        c, r = evaluate(solved.x)
        return solved.x, c, r @ r

    z, c, cost = refine(z0, 0)
    grid_resid = (obs - B @ grid_fit) / s
    if cost > grid_resid @ grid_resid + 1:
        return tau, grid_fit
    def smooth_spacing(z, c, cost):
        # Regularize log-time gaps only within the residual noise budget.
        if len(z) >= 3:
            for strength in (18, 10, 6, 3):
                zz, cc, err = refine(z, strength)
                if err <= cost + len(z) - 2:
                    return zz, cc
        return z, c

    full = smooth_spacing(z, c, cost)
    # If a smaller spectrum fits almost equally well, average the two
    # passive models instead of forcing a hard choice of model order.
    if len(z) >= 2:
        reduced = min((refine(np.delete(z, j), 0) for j in range(len(z))),
                      key=lambda model: model[2])
        if reduced[2] < cost + 1:
            zb, cb, loss = reduced
            if len(zb) >= 2:
                smaller = min((refine(np.delete(zb, j), 0) for j in range(len(zb))),
                              key=lambda model: model[2])
                if smaller[2] < loss + 1:
                    reduced = smaller
            zs, cs = smooth_spacing(*reduced)
            zf, cf = full
            return np.exp(np.r_[zf, zs]), 0.5 * np.r_[cf[0] + cs[0], cf[1:], cs[1:]]
    return np.exp(full[0]), full[1]


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Return clean test stress samples and the nonnegative equilibrium modulus."""
    t, strain, y, test_t, test_strain = (
        np.asarray(experiment[key], dtype=float)
        for key in ('train_t', 'train_strain', 'train_stress_raw', 'test_t', 'test_strain')
    )
    tau, c = _fit(t, strain, y)
    return {'test_stress': _design(test_t, test_strain, tau) @ c, 'G_inf': float(c[0])}
