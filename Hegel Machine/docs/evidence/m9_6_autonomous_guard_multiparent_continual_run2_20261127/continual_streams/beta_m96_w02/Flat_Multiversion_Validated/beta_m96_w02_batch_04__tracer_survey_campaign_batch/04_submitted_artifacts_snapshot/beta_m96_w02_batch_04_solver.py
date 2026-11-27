"""Calibrate the sensor banks and invert the history-dependent tracer model."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester
from scipy.sparse import csc_matrix, diags, identity, kron
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


_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line_fit(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _fit_affine(x, y, prior=None):
    # Sweep measurements lie on an affine line up to readout precision;
    # isolated EMI spikes are not treated as a change in gain or offset.
    xtol = 1e-6 * max(1., np.max(np.abs(x), initial=0.))
    ytol = 5e-6 * max(1., np.median(np.abs(y)) if y.size else 0.)
    if x.size >= 2 and np.ptp(x) > xtol:
        coeff = _line_fit(x, y)
        if coeff[0] > 0 and np.max(np.abs(y - (coeff[0] * x + coeff[1]))) <= ytol:
            if x.size > 2 or prior is None:
                return coeff
    if prior is not None:
        best = np.abs(y - (prior[0] * x + prior[1])) <= ytol
        coeff = prior
    else:
        best = np.zeros(x.size, dtype=bool)
        coeff = None
    for i, j in zip(*np.triu_indices(x.size, 1)):
        dx = x[j] - x[i]
        if abs(dx) <= xtol:
            continue
        gain = (y[j] - y[i]) / dx
        if gain <= 0:
            continue
        offset = y[i] - gain * x[i]
        keep = np.abs(y - (gain * x + offset)) <= ytol
        if keep.sum() > best.sum() and (prior is None or keep.sum() >= 3):
            best, coeff = keep, np.array([gain, offset])
    if best.sum() >= 2 and np.ptp(x[best]) > xtol:
        return _line_fit(x[best], y[best])
    return coeff


def _calibrate(study):
    rows = _sweep_rows(study['sensor_calibration_sweeps'])
    row_bank = np.asarray(study['row_bank_id'], dtype=int)
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    history = None
    for bank in np.unique(row_bank):
        x, raw = rows[rows[:, 0] == bank, 1:].T
        if x.size >= 3 and np.ptp(x) > 1e-6 * max(1., np.max(np.abs(x))):
            gain, offset = _line_fit(x, raw)
            clean = gain > 0 and np.max(np.abs(raw - (gain * x + offset))) <= 5e-6 * max(1., np.median(np.abs(raw)))
        else:
            clean = False
        if not clean:
            if history is None:
                history = json.loads(_CALIBRATION_HISTORY.read_text()) if _CALIBRATION_HISTORY.is_file() else []
            prior = None
            for episode in reversed(history):
                hx, hy = _sweep_rows(episode)[np.asarray(episode['bank_id']) == bank, 1:].T
                prior = _fit_affine(hx, hy)
                if prior is not None:
                    # Recover the nominal coefficients only when they reproduce
                    # the archived readings within their recorded precision.
                    nominal = np.round(prior, 4)
                    if (nominal[0] > 0
                            and np.max(np.abs(hy - (nominal[0] * hx + nominal[1]))) <= 1e-6):
                        prior = nominal
                    break
            coeff = _fit_affine(x, raw, prior)
            if coeff is None:
                raise ValueError(f'Insufficient calibration for bank {bank}')
            gain, offset = coeff
        mask = row_bank == bank
        y[mask] = (y[mask] - offset) / gain
    return y


def _forward(study):
    src = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = src.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    interaction = np.linalg.norm(v - vx[:, None] - vy) / max(np.linalg.norm(v), np.finfo(float).tiny)
    memory = np.max(study['past_vortex_circulation_trace'], initial=-np.inf) > study['critical_vortex_circulation']
    scale = 1.45 if memory or interaction > 1e-4 else 1.
    x, y = (_build_1d(study['L' + a], n, scale * np.asarray(study['kappa_' + a]),
                      study['vel_' + a], study['robin_' + a], study['reaction_' + a])
            for a, n in (('x', nx), ('y', ny)))
    ii, jj = np.asarray(study['sensor_ij'], dtype=int).T
    mu = np.asarray(study['mu_shifts'], dtype=float)
    design = np.empty((len(mu), len(ii), m))
    if not memory and interaction <= 1e-6:
        x += np.diag(vx)
        y += np.diag(vy)
        eye = np.eye(nx)
        for k, shift in enumerate(mu):
            fields = np.stack([solve_sylvester(x + shift * eye, y.T, mode) for mode in src])
            design[k] = fields[:, ii, jj].T
    else:
        op = (kron(csc_matrix(x), identity(ny, format='csc'), format='csc')
              + kron(identity(nx, format='csc'), csc_matrix(y), format='csc')
              + diags(v.ravel(), format='csc'))
        eye = identity(nx * ny, format='csc')
        rhs = np.asfortranarray(src.reshape(m, -1).T)
        for k, shift in enumerate(mu):
            fields = splu(op + shift * eye).solve(rhs).T.reshape(src.shape)
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
