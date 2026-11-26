"""Coupled finite-volume tracer inversion for the vortex survey."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

# Batch 07 clean sweeps updated Wing A. Its one clean check per Wing B
# bank corroborated the maps from the complete batch 03–04 sweeps.
_ARCHIVE_MAPS = (
    ((1.2246, 0.1292), (0.7697, -0.0964), (1.0, 0.0)),
    ((0.6813, -0.1532), (1.1935, 0.1420), (1.0, 0.0)),
    ((0.7170, -0.0972), (1.0, 0.0)),
    ((1.1920, 0.0807), (1.0, 0.0)),
)


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
    axes = [_build_1d(study['L' + a], n, 1.45 * np.asarray(study['kappa_' + a]),
                      np.asarray(study['vel_' + a]), study['robin_' + a],
                      np.asarray(study['reaction_' + a]))
            for a, n in (('x', nx), ('y', ny))]
    base = (kron(csc_matrix(axes[0]), eye(ny, format='csc'), format='csc')
            + kron(eye(nx, format='csc'), csc_matrix(axes[1]), format='csc')
            + diags(np.asarray(study['V_reaction_2d']).ravel(), format='csc'))
    identity = eye(nx * ny, format='csc')
    rhs = np.asfortranarray(source.reshape(m, -1).T)
    ij = np.asarray(study['sensor_ij'], dtype=int)
    indices = ij[:, 0] * ny + ij[:, 1]
    observations = []
    for shift in study['mu_shifts']:
        field = splu(base + shift * identity).solve(rhs)
        observations.append(field[indices])
    return np.vstack(observations), field.reshape(nx, ny, m)


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


def _calibration(study, design):
    sweep = study['sensor_calibration_sweeps']
    banks, x, y = (np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    maps = []
    ambiguous = {}
    tol = 2e-6
    for b, history in enumerate(_ARCHIVE_MAPS):
        xb, yb = x[banks == b], y[banks == b]
        maps.append(_bank_fit(xb, yb, history))
        if xb.size != 2 or np.ptp(xb) <= tol:
            continue
        candidates = np.asarray(history)
        errors = np.abs(xb[:, None] * candidates[:, 0] + candidates[:, 1] - yb[:, None])
        matches = (errors <= tol).sum(axis=0)
        g = (yb[1] - yb[0]) / (xb[1] - xb[0])
        if g > 0 and matches.max() < 2:
            maps[-1] = candidates[np.argmax(matches)]
            ambiguous[b] = np.array([g, yb[0] - g * xb[0]])
    maps = np.asarray(maps)
    if not ambiguous:
        return maps

    # Two readings alone cannot distinguish an affine change from two spikes.
    # Fit the survey without the ambiguous banks and score their predictions
    # in the common raw-readout units, regardless of candidate gain.
    row_bank = np.asarray(study['row_bank_id'], dtype=int)
    keep = ~np.isin(row_bank, list(ambiguous))
    if not keep.any():
        return maps
    raw = np.asarray(study['y_obs_raw'])
    z = np.asarray(study['Z_nuisance'])
    ridge = study['nuisance_ridge']
    g, o = maps[row_bank[keep]].T
    target = (raw[keep] - o) / g
    profiled = _profile(z[keep], np.column_stack((design[keep], target)), ridge)
    w, _ = _box_simplex(profiled[:, :-1], profiled[:, -1], study['source_ridge'],
                        study['u_box'], study['total_budget'])
    z_fit = np.vstack((z[keep], np.sqrt(ridge) * np.eye(z.shape[1])))
    residual = np.r_[target - design[keep] @ w, np.zeros(z.shape[1])]
    nuisance = np.linalg.lstsq(z_fit, residual, rcond=None)[0]
    prediction = design @ w + z @ nuisance
    for b, fit in ambiguous.items():
        rows = row_bank == b
        archived = maps[b]
        err_fit = raw[rows] - (fit[0] * prediction[rows] + fit[1])
        err_archive = raw[rows] - (archived[0] * prediction[rows] + archived[1])
        if err_fit @ err_fit < err_archive @ err_archive:
            maps[b] = fit
    return maps


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


def solve_beta_w01_batch_08(study: dict) -> dict:
    design, terminal = _responses(study)
    gain, offset = _calibration(study, design)[np.asarray(study['row_bank_id'], dtype=int)].T
    y = (np.asarray(study['y_obs_raw']) - offset) / gain
    values = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)),
                      study['nuisance_ridge'])
    w, cert = _box_simplex(values[:, :-1], values[:, -1], study['source_ridge'],
                           study['u_box'], study['total_budget'])
    return dict(weights=w, terminal_field=terminal @ w, kkt_certificate=cert)
