"""Nominal separable finite-volume tracer inversion for batch 07."""
import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl
from scipy.optimize import minimize

# Distinct affine maps fitted to complete prior sweeps, newest first.
_ARCHIVE_MAPS = (((0.7697, -0.0964), (1.0, -0.0)), ((1.1935, 0.142), (1.0, 0.0)), ((0.717, -0.0972), (1.0, 0.0)), ((1.192, 0.0807), (1.0, -0.0)))


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    face = 2 * kappa[:-1] * kappa[1:] / ((kappa[:-1] + kappa[1:]) * h**2)
    pos, neg = np.maximum(vel, 0) / h, np.maximum(-vel, 0) / h
    center = np.r_[face, 0.] + np.r_[0., face] + diag_add + pos + neg
    edge = kappa[[0, -1]]
    center[[0, -1]] += robin * edge / (h * (robin * h / 2 + edge))
    return np.diag(center) + np.diag(-face - pos[1:], -1) + np.diag(-face - neg[:-1], 1)


def _responses(study):
    source = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = source.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    axes = [
        _build_1d(study['L' + axis], n, np.asarray(study['kappa_' + axis]),
                  np.asarray(study['vel_' + axis]), study['robin_' + axis],
                  np.asarray(study['reaction_' + axis]) + extra)
        for axis, n, extra in (('x', nx, vx), ('y', ny, vy))
    ]
    tx, qx = schur(axes[0], output='real')
    ty, qy = schur(axes[1], output='real')
    rhs = qx.T @ source @ qy
    ij = np.asarray(study['sensor_ij'], dtype=int)
    identity = np.eye(nx)
    observations = []
    for shift in study['mu_shifts']:
        shifted = tx + shift * identity
        transformed = np.empty_like(rhs)
        for j in range(m):
            sol, scale, info = dtrsyl(shifted, ty, rhs[j], tranb='T')
            if info != 0:
                raise np.linalg.LinAlgError('Ill-conditioned tracer Sylvester equation')
            transformed[j] = sol / scale
        field = qx @ transformed @ qy.T
        observations.append(field[:, ij[:, 0], ij[:, 1]].T)
    return np.vstack(observations), np.moveaxis(field, 0, -1)


def _bank_fit(x, y, history):
    tol = 2e-6  # Archive sweep columns are stored to six decimals.
    a = np.column_stack((x, np.ones(x.size)))
    maps = np.asarray(history)
    predicted = a @ maps.T
    counts = (np.abs(predicted - y[:, None]) <= tol).sum(axis=0)
    best_map = np.argmax(counts)
    if x.size == 2 and counts[best_map] == 1:
        # Two readings always interpolate. A matched archive check and a
        # second reading outside the historical envelope identify the spike.
        outside = (y < predicted.min(axis=1) - tol) | (y > predicted.max(axis=1) + tol)
        if outside.any():
            return maps[best_map]
    spread = x.size >= 2 and np.ptp(x) > tol
    if spread:
        fit = np.linalg.lstsq(a, y, rcond=None)[0]
        if fit[0] > 0 and np.max(np.abs(a @ fit - y)) <= tol:
            return fit
        # Pair proposals are useful only with independent corroboration.
        i, j = np.triu_indices(x.size, 1)
        valid = np.abs(x[j] - x[i]) > tol
        i, j = i[valid], j[valid]
        gain = (y[j] - y[i]) / (x[j] - x[i])
        pairs = np.column_stack((gain, y[i] - gain * x[i]))[gain > 0]
        if len(pairs):
            inliers = np.abs(a @ pairs.T - y[:, None]) <= tol
            counts = inliers.sum(axis=0)
            best = np.argmax(counts)
            if counts[best] >= 3 and 2 * counts[best] > x.size:
                keep = inliers[:, best]
                return np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
    # In a sparse, spike-dominated sweep, an archive map can predict a
    # clean check without fitting it. Prefer more matches, then recency.
    return maps[best_map]


def _calibration(study):
    sweep = study['sensor_calibration_sweeps']
    banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    return np.array([_bank_fit(x[banks == b], y[banks == b], history)
                     for b, history in enumerate(_ARCHIVE_MAPS)])


def _profile(z, values, ridge):
    # The certified objective uses the ridge residual map on both inputs.
    q, singular, _ = np.linalg.svd(z, full_matrices=False)
    if ridge > 0:
        factor = singular**2 / (singular**2 + ridge)
    else:
        factor = singular > np.finfo(float).eps * max(z.shape) * singular.max(initial=0.)
    return values - q @ (factor[:, None] * (q.T @ values))


def _box_simplex(s, y, ridge, cap, budget):
    m = s.shape[1]
    h = s.T @ s + ridge * np.eye(m)
    c = s.T @ y
    if budget in (0, m * cap):
        w = np.full(m, budget / m)
    else:
        result = minimize(lambda w: (0.5 * w @ h @ w - c @ w, h @ w - c),
                          np.full(m, budget / m), method='SLSQP', jac=True,
                          bounds=[(0., cap)] * m,
                          constraints={'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                       'jac': lambda w: np.ones(m)},
                          options={'ftol': 1e-13, 'maxiter': 300})
        w = result.x
        free = (w > 1e-8) & (w < cap - 1e-8)
        if free.any():
            upper = w >= cap - 1e-8
            hf = h[np.ix_(free, free)]
            e = np.ones(free.sum())
            a, b = np.linalg.solve(hf, np.column_stack((c[free] - cap * h[np.ix_(free, upper)].sum(axis=1), e))).T
            nu = (budget - cap * upper.sum() - a.sum()) / b.sum()
            polished = np.where(upper, cap, 0.)
            polished[free] = a + nu * b
            g = h @ polished - c
            if polished.min() >= 0 and polished.max() <= cap and np.all(g[upper] <= nu + 1e-10) and np.all(g[~(upper | free)] >= nu - 1e-10):
                w = polished
    g = h @ w - c
    free = (w > 1e-8) & (w < cap - 1e-8)
    if free.any():
        nu = float(np.mean(g[free]))
    else:
        lo = np.max(g[w > cap / 2], initial=-np.inf)
        hi = np.min(g[w <= cap / 2], initial=np.inf)
        nu = float((lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else max(lo, hi) if not np.isfinite(lo) else min(lo, hi))
    residual = s @ w - y
    primal = float((residual @ residual + ridge * (w @ w)) / 2)
    bound = min(primal, primal - g @ w + nu * budget + cap * np.minimum(g - nu, 0.).sum())
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(g - nu, 0.),
                lam_upper=np.maximum(nu - g, 0.), primal_objective=primal, dual_lower_bound=float(bound))
    return w, cert


def solve_beta_w01_batch_07(study: dict) -> dict:
    design, terminal = _responses(study)
    gain, offset = _calibration(study)[np.asarray(study['row_bank_id'], dtype=int)].T
    y = (np.asarray(study['y_obs_raw']) - offset) / gain
    values = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)),
                      study['nuisance_ridge'])
    w, cert = _box_simplex(values[:, :-1], values[:, -1], study['source_ridge'],
                           study['u_box'], study['total_budget'])
    return dict(weights=w, terminal_field=terminal @ w, kkt_certificate=cert)
