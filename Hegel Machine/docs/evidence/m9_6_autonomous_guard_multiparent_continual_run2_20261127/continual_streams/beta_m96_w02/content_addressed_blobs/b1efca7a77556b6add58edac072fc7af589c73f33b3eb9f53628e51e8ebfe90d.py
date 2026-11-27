"""Recover tracer source weights and the field at the final reaction shift."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    kappa = np.asarray(kappa, dtype=float)
    vel = np.asarray(vel, dtype=float)
    h = L / n
    face = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    pos = np.maximum(vel, 0) / h
    neg = np.maximum(-vel, 0) / h
    diag = np.asarray(diag_add, dtype=float) + np.r_[0, face] + np.r_[face, 0] + pos + neg
    for b in (0, n - 1):
        diag[b] += robin * kappa[b] / (h * (robin * h / 2 + kappa[b]))
    return np.diag(diag) - np.diag(face + pos[1:], -1) - np.diag(face + neg[:-1], 1)


def _sweep_rows(sweeps):
    batches = [sweeps] if isinstance(sweeps, dict) else sweeps
    cols = []
    for d in batches:
        arrs = np.broadcast_arrays(d['bank_id'], d['ref_signal'], d['raw_readout'])
        cols.extend(zip(*(a.ravel() for a in arrs)))
    return np.asarray(cols, dtype=float).reshape(-1, 3)


def _fit_affine(x, y):
    """Use a high-breakdown seed to keep isolated EMI spikes out of OLS."""
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    good = np.abs(dx) > 1e-10 * max(1., np.max(np.abs(x), initial=0.))
    if not np.any(good):
        return None
    gain = np.median((y[j[good]] - y[i[good]]) / dx[good])
    residual = y - gain * x
    residual -= np.median(residual)
    sigma = 1.4826 * np.median(np.abs(residual))
    keep = np.abs(residual) <= max(5 * sigma, 1e-10 * max(1., np.max(np.abs(y))))
    if np.ptp(x[keep]) <= 1e-10 * max(1., np.max(np.abs(x))):
        return None
    return np.linalg.lstsq(np.column_stack((x[keep], np.ones(np.count_nonzero(keep)))), y[keep], rcond=None)[0]


_CALIBRATION_HISTORY = Path("/tmp/hegel_continual_archive/archive_calibration_history.json")


def _calibrate(study):
    rows = _sweep_rows(study['sensor_calibration_sweeps'])
    banks = np.asarray(study['row_bank_id'], dtype=int)
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    calibrated = np.empty_like(raw)
    history = None
    for b in np.unique(banks):
        x, y = rows[rows[:, 0] == b, 1:].T
        coeff = _fit_affine(x, y) if x.size >= 3 else None
        if x.size == 3 and coeff is not None:
            # Three non-collinear checks cannot identify which reading spiked.
            # Use the latest usable prefix rather than choosing an arbitrary pair.
            if np.max(np.abs(y - (coeff[0] * x + coeff[1]))) > 1e-5:
                coeff = None
        if coeff is None or coeff[0] <= 0:
            if history is None:
                history = json.loads(_CALIBRATION_HISTORY.read_text())
            for sweep in reversed(history):
                previous = _sweep_rows(sweep)
                px, py = previous[previous[:, 0] == b, 1:].T
                prior = _fit_affine(px, py)
                if prior is not None and prior[0] > 0:
                    coeff = prior
                    break
            if coeff is None or coeff[0] <= 0:
                raise ValueError(f'No identifiable calibration for bank {b}')
            # Two consistent, separated checks can resolve a small current-batch drift.
            if x.size == 2 and np.ptp(x) > 1e-8:
                if np.max(np.abs(y - (coeff[0] * x + coeff[1]))) < 1e-5:
                    current = _fit_affine(x, y)
                    if current is not None and current[0] > 0:
                        coeff = current
        mask = banks == b
        calibrated[mask] = (raw[mask] - coeff[1]) / coeff[0]
    return calibrated


def _forward(study):
    src = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = src.shape
    V = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = V.mean(axis=1)
    vy = V.mean(axis=0) - V.mean()
    norm = np.linalg.norm(V)
    nonsep = np.linalg.norm(V - vx[:, None] - vy) / norm if norm else 0.
    eddy = nonsep > 1e-4 or np.any(
        np.asarray(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation'])
    scale = 1.45 if eddy else 1.0
    X = _build_1d(study['Lx'], nx, scale * np.asarray(study['kappa_x']),
                  study['vel_x'], study['robin_x'], study['reaction_x'])
    Y = _build_1d(study['Ly'], ny, scale * np.asarray(study['kappa_y']),
                  study['vel_y'], study['robin_y'], study['reaction_y'])
    ii, jj = np.asarray(study['sensor_ij'], dtype=int).T
    mu = np.asarray(study['mu_shifts'], dtype=float)
    design = np.empty((len(mu), len(ii), m))
    if not eddy and nonsep <= 1e-6:
        X += np.diag(vx)
        Y += np.diag(vy)
        identity = np.eye(nx)
        for k, shift in enumerate(mu):
            shifted = X + shift * identity
            fields = np.stack([solve_sylvester(shifted, Y.T, source) for source in src])
            design[k] = fields[:, ii, jj].T
    else:
        A = (kron(csc_matrix(X), eye(ny), format='csc')
             + kron(eye(nx), csc_matrix(Y), format='csc')
             + diags(V.ravel(), format='csc'))
        identity = eye(nx * ny, format='csc')
        rhs = src.reshape(m, -1).T
        for k, shift in enumerate(mu):
            fields = splu(A + shift * identity).solve(rhs).T.reshape(m, nx, ny)
            design[k] = fields[:, ii, jj].T
    return design.reshape(-1, m), fields


def _profile(Z, values, ridge):
    # The observation contract uses the ridge residual map P, not sqrt(P).
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    if ridge == 0:
        factor = (s > np.finfo(float).eps * max(Z.shape) * np.max(s, initial=0.)).astype(float)
    else:
        factor = s**2 / (s**2 + ridge)
    return values - U @ (factor[:, None] * (U.T @ values))


def _box_simplex(S, y, ridge, upper, budget):
    n = S.shape[1]
    if budget < 0 or budget > n * upper:
        raise ValueError('Infeasible source budget')
    H = S.T @ S + ridge * np.eye(n)
    q = -(S.T @ y)
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * upper:
        w.fill(0. if budget == 0 else upper)
        g = H @ w + q
        return w, float(np.min(g) if budget == 0 else np.max(g))

    # Active bounds: -1 at zero, 0 free, +1 at the upper cap.
    state = np.zeros(n, dtype=int)
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        fixed = np.flatnonzero(state != 0)
        if not len(free):
            g = H @ w + q
            low, high = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = low[np.argmin(g[low])], high[np.argmax(g[high])]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            state[[i, j]] = 0
            continue
        k = len(free)
        system = np.empty((k + 1, k + 1))
        system[:k, :k] = H[np.ix_(free, free)]
        system[:k, k] = -1
        system[k, :k] = 1
        system[k, k] = 0
        rhs = np.r_[-q[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        try:
            sol = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(system, rhs, rcond=None)[0]
        target, nu = sol[:-1], sol[-1]
        step = target - w[free]
        hit = np.flatnonzero((target < -1e-12) | (target > upper + 1e-12))
        if len(hit):
            bounds = np.where(step[hit] < 0, 0., upper)
            fractions = (bounds - w[free[hit]]) / step[hit]
            p = np.argmin(fractions)
            w[free] += np.clip(fractions[p], 0., 1.) * step
            idx = free[hit[p]]
            w[idx] = bounds[p]
            state[idx] = -1 if bounds[p] == 0 else 1
            continue
        w[free] = np.clip(target, 0., upper)
        violation = state * (H @ w + q - nu)
        j = np.argmax(violation)
        if violation[j] <= 1e-12:
            return w, float(nu)
        state[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m96_w02_batch_04(study: dict) -> dict:
    y = _calibrate(study)
    obs, terminal_modes = _forward(study)
    joined = _profile(np.asarray(study['Z_nuisance'], dtype=float), np.column_stack((obs, y)), study['nuisance_ridge'])
    S, r = joined[:, :-1], joined[:, -1]
    ridge, upper, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, r, ridge, upper, budget)
    residual = S @ w - r
    g = S.T @ residual + ridge * w
    reduced = g - nu
    low, high = np.maximum(reduced, 0.), np.maximum(-reduced, 0.)
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    lower = primal - low @ w - high @ (upper - w) + nu * (budget - w.sum())
    return {
        'weights': w,
        'terminal_field': np.einsum('m,mij->ij', w, terminal_modes),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': low.tolist(),
            'lam_upper': high.tolist(), 'primal_objective': float(primal),
            'dual_lower_bound': float(min(primal, lower)),
        },
    }
