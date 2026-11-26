"""Calibrated multi-wing finite-volume tracer inversion."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    face = 2 * kappa[:-1] * kappa[1:] / ((kappa[:-1] + kappa[1:]) * h**2)
    pos, neg = np.maximum(vel, 0) / h, np.maximum(-vel, 0) / h
    center = np.r_[face, 0.] + np.r_[0., face] + diag_add + pos + neg
    edge = kappa[[0, -1]]
    center[[0, -1]] += robin * edge / (h * (robin * h / 2 + edge))
    return diags((-face - pos[1:], center, -face - neg[:-1]), (-1, 0, 1), format='csc')


def _responses(study):
    source = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = source.shape
    axes = []
    for axis, n in zip('xy', (nx, ny)):
        axes.append(_build_1d(study['L' + axis], n, 1.45 * np.asarray(study['kappa_' + axis]),
                              np.asarray(study['vel_' + axis]), study['robin_' + axis],
                              np.asarray(study['reaction_' + axis])))
    a = kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc')
    v = np.asarray(study['V_reaction_2d']).ravel()
    rhs = source.reshape(m, -1).T
    sensors = np.asarray(study['sensor_ij'], dtype=int)
    sensor_rows = sensors[:, 0] * ny + sensors[:, 1]
    observations = []
    for shift in study['mu_shifts']:
        field = splu(a + diags(v + shift, format='csc')).solve(rhs)
        observations.append(field[sensor_rows])
    return np.vstack(observations), field.reshape(nx, ny, m)


def _sweep_columns(sweep):
    return tuple(np.asarray(sweep[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))


def _fit_affine(x, y):
    if x.size < 2 or np.ptp(x) < 1e-6:
        return None
    design = np.column_stack((x, np.ones(x.size)))
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    if x.size < 4 or np.max(np.abs(design @ fit - y)) <= 1e-6:
        return fit if fit[0] > 0 else None
    # Least-trimmed initialization avoids giving leverage to a single EMI spike.
    i, j = np.triu_indices(x.size, 1)
    good = np.abs(x[j] - x[i]) > 1e-6
    i, j = i[good], j[good]
    gain = (y[j] - y[i]) / (x[j] - x[i])
    good = gain > 0
    gain, i = gain[good], i[good]
    if not gain.size:
        return None
    candidates = np.column_stack((gain, y[i] - gain * x[i]))
    errors = (design @ candidates.T - y[:, None])**2
    keep = (x.size + 3) // 2
    fit = candidates[np.argmin(np.sum(np.partition(errors, keep - 1, axis=0)[:keep], axis=0))]
    residual = y - design @ fit
    median = np.median(residual)
    scale = 1.4826 * np.median(np.abs(residual - median))
    inliers = np.abs(residual - median) <= max(1e-6, 4.5 * scale)
    fit = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    return fit if fit[0] > 0 else None


def _calibration(study):
    banks, x, y = _sweep_columns(study['sensor_calibration_sweeps'])
    fits = [_fit_affine(x[banks == b], y[banks == b]) for b in range(4)]
    counts = np.bincount(banks.astype(int), minlength=4)
    if any(f is None for f in fits) or np.any((counts >= 2) & (counts <= 3)):
        archive = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        history = json.loads(archive.read_text()) if archive.is_file() else []
        for b, fit in enumerate(fits):
            if fit is not None and not (2 <= counts[b] <= 3):
                continue
            prior = None
            for entry in reversed(history):
                bank_h, x_h, y_h = _sweep_columns(entry)
                prior = _fit_affine(x_h[bank_h == b], y_h[bank_h == b])
                if prior is not None:
                    break
            if fit is not None and prior is None:
                continue
            gain, offset = prior if prior is not None else (1., 0.)
            local = banks == b
            offsets = y[local] - gain * x[local]
            if 2 <= counts[b] <= 3 and prior is not None:
                # Too few points to reject a spike from the local fit alone.
                # Use history only when enough current points corroborate it.
                agrees = np.abs(offsets - offset) <= 2e-6
                if (counts[b] + 1) // 2 <= agrees.sum() < counts[b]:
                    fits[b] = (gain, np.median(offsets[agrees]))
                    continue
            if fit is None:
                fits[b] = (gain, np.median(offsets) if offsets.size else offset)
    return np.asarray(fits)


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


def solve_beta_w01_batch_04(study: dict) -> dict:
    design, terminal = _responses(study)
    gain, offset = _calibration(study)[np.asarray(study['row_bank_id'], dtype=int)].T
    y = (np.asarray(study['y_obs_raw']) - offset) / gain
    values = _profile(np.asarray(study['Z_nuisance']), np.column_stack((design, y)), study['nuisance_ridge'])
    w, cert = _box_simplex(values[:, :-1], values[:, -1], study['source_ridge'], study['u_box'], study['total_budget'])
    return dict(weights=w, terminal_field=terminal @ w, kkt_certificate=cert)
