"""Identify a nonnegative Prony spectrum from a strain-controlled experiment."""

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter


def _history(t, strain):
    return np.diff(t), np.diff(strain), strain


def _basis(history, tau):
    dt, de, strain = history
    a = np.empty((len(strain), len(tau) + 1))
    a[:, 0] = strain
    a[0, 1:] = 0.
    if not len(dt) or not len(tau):
        return a
    # Exact convolution for a linear strain ramp between observations.
    if np.max(abs(dt - dt[0])) <= 1e-10 * dt[0]:
        z = dt[0] / tau
        for j, (q, w) in enumerate(zip(np.exp(-z), -np.expm1(-z) / z), 1):
            a[1:, j] = lfilter([w], [1., -q], de)
    else:
        z = dt[:, None] / tau
        q = np.exp(-z)
        u = de[:, None] * (-np.expm1(-z) / z)
        for i in range(len(dt)):
            a[i + 1, 1:] = q[i] * a[i, 1:] + u[i]
    return a


def _mad(x):
    return max(1.4826 * np.median(abs(x - np.median(x))),
               np.finfo(float).eps)


def _project(q, x):
    return x - np.einsum('ij,j...->i...', q, np.einsum('ij,i...->j...', q, x))


def _pilot(a, d, y):
    weight = np.ones(len(y))
    for _ in range(24):
        q, _ = np.linalg.qr(d * weight[:, None], mode='reduced')
        ap = _project(q, a * weight[:, None])
        yp = _project(q, y * weight)
        norm = np.linalg.norm(ap, axis=0)
        norm[norm < np.finfo(float).eps] = 1.
        c = nnls(ap / norm, yp, maxiter=2000)[0] / norm
        r = y - np.einsum('ij,j->i', a, c)
        r -= d @ np.linalg.lstsq(d * weight[:, None], r * weight, rcond=None)[0]
        r -= np.median(r)
        s = _mad(r)
        new_weight = 1. / np.hypot(1., r / (3. * s))
        if np.max(abs(new_weight - weight)) < 1e-5:
            break
        weight = new_weight
    return c, r, s


def _clusters(grid, mass, dt, n):
    keep = mass > max(np.max(mass) * 1e-5, 1e-12)
    z = list(np.log(grid[keep]))
    w = list(mass[keep])
    while len(z) > n:
        cost = [w[i] * w[i + 1] / (w[i] + w[i + 1]) * (z[i + 1] - z[i])**2
                for i in range(len(z) - 1)]
        i = np.argmin(cost)
        z[i:i + 2] = [(w[i] * z[i] + w[i + 1] * z[i + 1]) / (w[i] + w[i + 1])]
        w[i:i + 2] = [w[i] + w[i + 1]]
    generic = np.linspace(np.log(max(.02, min(dt, .15))), np.log(12.), n + 2)[1:-1]
    return (np.asarray(z) if len(z) == n else generic), generic


def _spectrum(t, e, y, penalty=.5):
    import os
    order_cost = float(os.environ.get("PRONY_ORDER_COST", "4"))
    soft_order = int(os.environ.get("PRONY_SOFT_ORDER", "0"))
    gate = int(os.environ.get("PRONY_GATE", "1"))
    elapsed = t - t[0]
    duration = elapsed[-1]
    history = _history(t, e)
    drift = np.column_stack((np.ones(len(t)), elapsed / duration - .5))
    grid = np.geomspace(.02, 20., 80)
    a = _basis(history, grid)
    c, residual, scale = _pilot(a, drift, y)
    spike = abs(residual) > 6. * scale
    good = ~spike
    target = y
    regressor = None
    # Reuse spike samples only when the data support a common signed jump size.
    if np.count_nonzero(spike) >= 6:
        magnitude = np.median(abs(residual[spike]))
        if magnitude > 30. * scale and _mad(abs(residual[spike])) < 2. * scale:
            candidate = spike & (abs(abs(residual) - magnitude) < 4. * scale)
            good = good | candidate
            target = y.copy()
            target[candidate] -= np.sign(residual[candidate]) * magnitude
            regressor = np.where(candidate, np.sign(residual), 0.)

    if regressor is not None:
        drift = np.column_stack((drift, regressor))
    q, _ = np.linalg.qr(drift[good], mode='reduced')
    yp = _project(q, target[good])
    mass = c[1:] * np.linalg.norm(_project(q, a[good, 1:]), axis=0)
    nobs = len(yp)
    basis0 = _project(q, e[good])
    norm0 = basis0 @ basis0
    g0 = max(0., basis0 @ yp / norm0) if norm0 else 0.
    best = (nobs * np.log(max(np.mean((basis0 * g0 - yp)**2), scale**2 * 1e-6)),
            np.array([g0]), np.empty(0))

    models = [best]

    # Compare model orders without a time prior. Refit the selected order with
    # a weak penalty for extrapolating relaxation beyond the record duration.
    peak_count = np.count_nonzero((mass > max(np.max(mass) * 1e-5, 1e-12)) &
                                  (np.r_[True, mass[:-1] == 0]))
    max_order = min(7, max(3, peak_count + 1), max(0, (nobs - 4) // 3))
    lnlow, lnhigh = np.log([.02, 20.])
    bounds = None
    for m in range(1, max_order + 1):
        first, second = _clusters(grid, mass, np.median(history[0]), m)
        bounds = (np.r_[np.zeros(m + 1), np.full(m, lnlow)],
                  np.r_[np.full(m + 1, np.inf), np.full(m, lnhigh)])

        def fun(v):
            ap = _project(q, _basis(history, np.exp(v[m + 1:]))[good])
            return (ap @ v[:m + 1] - yp) / scale

        best_m = None
        for start in (first, second) if not np.allclose(first, second) else (first,):
            ap = _project(q, _basis(history, np.exp(start))[good])
            coef = nnls(ap, yp)[0]
            f = least_squares(fun, np.r_[coef, start], bounds=bounds,
                              x_scale='jac', max_nfev=120, ftol=1e-8)
            mse = np.mean(f.fun**2) * scale**2
            score = nobs * np.log(max(mse, scale**2 * 1e-6)) + m * order_cost
            if best_m is None or score < best_m[0]:
                best_m = score, f.x[:m + 1], f.x[m + 1:]
            if score < best[0]:
                best = score, f.x[:m + 1], f.x[m + 1:]

        models.append(best_m)

    def refit(model):
        _, coef, logtau = model
        m = len(logtau)
        local_penalty = penalty if not gate or (m and np.max(np.exp(logtau)) <= duration * .4) else 0.
        if m and local_penalty:
            bounds = (np.r_[np.zeros(m + 1), np.full(m, lnlow)],
                      np.r_[np.full(m + 1, np.inf), np.full(m, lnhigh)])

            def regularized(v):
                tau = np.exp(v[m + 1:])
                ap = _project(q, _basis(history, tau)[good])
                return np.r_[(ap @ v[:m + 1] - yp) / scale,
                             np.sqrt(local_penalty) * 3. * tau / duration]

            v = least_squares(regularized, np.r_[coef, logtau], bounds=bounds,
                              x_scale='jac', max_nfev=120, ftol=1e-8).x
            coef, logtau = v[:m + 1], v[m + 1:]
        return coef, np.exp(logtau)
    if not soft_order:
        return refit(best)
    models = [model for model in models if model[0] - best[0] < 12.]
    weights = np.exp(-.5 * np.array([model[0] - best[0] for model in models]))
    weights /= weights.sum()
    def profile(model):
        coef, tau = refit(model)
        m = len(tau)
        # If the first memory is only resolved over a few samples, integrate
        # plausible fast times instead of extrapolating a likelihood maximizer.
        order = np.argsort(tau)
        tau, coef = tau[order], np.r_[coef[0], coef[1:][order]]
        cutoff = 3. * np.median(history[0])
        if not m or tau[0] > cutoff:
            return coef, tau
        logtau = np.log(tau)
        limit = min(lnhigh, np.log(2. * cutoff), logtau[1] if m > 1 else lnhigh)
        if limit <= lnlow + .01:
            return coef, tau
        step = (limit - lnlow) / 11.
        probes = lnlow + step * (np.arange(11) + .5)
        local_penalty = penalty if not gate or tau[-1] <= .4 * duration else 0.
        costs, cs, ts = [], [], []
        for z in probes:
            def residual(v):
                qtau = np.exp(np.r_[z, v[m + 1:]])
                ap = _project(q, _basis(history, qtau)[good])
                rr = (ap @ v[:m + 1] - yp) / scale
                return np.r_[rr, np.sqrt(local_penalty) * 3. * qtau / duration] if local_penalty else rr
            lower = np.r_[np.zeros(m + 1), np.full(m-1, z)]
            upper = np.r_[np.full(m + 1, np.inf), np.full(m-1, lnhigh)]
            f = least_squares(residual, np.r_[coef, np.maximum(logtau[1:], z)],
                              bounds=(lower, upper), x_scale='jac', max_nfev=45, ftol=1e-7)
            costs.append(f.cost)
            cs.append(f.x[:m + 1]); ts.append(np.exp(np.r_[z, f.x[m + 1:]]))
        # Fine likelihood peaks cannot be integrated on this coarse grid.
        w = np.exp(np.min(costs) - np.asarray(costs)); w /= w.sum()
        if np.max(w) > .65:
            return coef, tau
        return np.r_[sum(wi*c[0] for wi,c in zip(w,cs)), np.concatenate([wi*c[1:] for wi,c in zip(w,cs)])], np.concatenate(ts)

    fitted = [profile(model) for model in models]
    g = sum(w*c[0] for w, (c, tau) in zip(weights, fitted))
    return (np.concatenate(([g], *(w*c[1:] for w, (c, tau) in zip(weights, fitted)))),
            np.concatenate([tau for _, tau in fitted]))


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Return clean test stress and the equilibrium modulus inferred from training."""
    t, e, y, tt, et = (np.asarray(experiment[k], dtype=float) for k in
                       ('train_t', 'train_strain', 'train_stress_raw',
                        'test_t', 'test_strain'))
    coef, tau = _spectrum(t, e, y)
    return {'test_stress': _basis(_history(tt, et), tau) @ coef,
            'G_inf': float(coef[0])}
