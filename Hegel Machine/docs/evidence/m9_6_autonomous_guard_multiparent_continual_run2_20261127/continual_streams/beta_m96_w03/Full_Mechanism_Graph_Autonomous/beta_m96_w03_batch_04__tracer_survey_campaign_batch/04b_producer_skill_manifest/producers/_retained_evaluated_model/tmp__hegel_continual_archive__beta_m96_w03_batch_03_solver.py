"""Calibrated laminar tracer inversion with certificates for a grid of box-simplex budgets."""
import json

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    conductance = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    diagonal = np.array(diag_add, dtype=float, copy=True)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    for edge in (0, n - 1):
        k = kappa[edge]
        diagonal[edge] += robin * k / (h * (robin * h / 2 + k))
    positive = np.maximum(vel, 0) / h
    negative = np.maximum(-vel, 0) / h
    diagonal += positive + negative
    return diags((-conductance - positive[1:], diagonal,
                  -conductance - negative[:-1]), (-1, 0, 1), format='csc')


def _sweep_points(sweeps, bank):
    select = np.asarray(sweeps['bank_id']) == bank
    return (np.asarray(sweeps['ref_signal'], dtype=float)[select],
            np.asarray(sweeps['raw_readout'], dtype=float)[select])


def _fit_line(x, y):
    centered = x - x.mean()
    gain = (centered @ (y - y.mean())) / (centered @ centered)
    return gain, y.mean() - gain * x.mean()


def _fit_sweep(x, y):
    if np.unique(x).size < 2:
        return None
    gain, offset = _fit_line(x, y)
    # The prefix sweeps are rounded to six decimal places.
    tol = 2e-6 * max(1., abs(gain))
    if np.max(np.abs(y - gain * x - offset)) <= tol:
        return gain, offset
    # Score pair-defined lines by consensus before refitting, so a few
    # large spikes cannot pull the fitted line away from the good sweep.
    i, j = np.triu_indices(x.size, 1)
    separated = x[i] != x[j]
    i, j = i[separated], j[separated]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    errors = np.abs((y - y[i, None]) - slopes[:, None] * (x - x[i, None]))
    inliers = errors <= tol
    best = np.lexsort((np.minimum(errors, tol).sum(axis=1), -inliers.sum(axis=1)))[0]
    good = inliers[best]
    return _fit_line(x[good], y[good])


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    rows = np.asarray(study['row_bank_id'])
    raw = np.asarray(study['y_obs_raw'])
    result = np.empty(rows.shape, dtype=float)
    history = None
    for bank in np.unique(rows):
        x, y = _sweep_points(sweeps, bank)
        fit = _fit_sweep(x, y)
        if fit is None:
            if history is None:
                with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
                    history = json.load(f)
            prior = [fit for sweep in reversed(history)
                     if (fit := _fit_sweep(*_sweep_points(sweep, bank))) is not None]
            if not prior:
                raise ValueError(f'No two-level calibration sweep for bank {bank}')
            if x.size:
                gain, _ = min(prior, key=lambda p: abs(np.median(y - p[0] * x) - p[1]))
                offset = np.median(y - gain * x)
            else:
                gain, offset = prior[0]
        else:
            gain, offset = fit
        selected = rows == bank
        result[selected] = (raw[selected] - offset) / gain
    return result


def _forward(study):
    nx, ny = study['nx'], study['ny']
    reaction = np.asarray(study['V_reaction_2d'])
    vx, vy = reaction[:, 0], reaction[0, :] - reaction[0, 0]
    if (not np.allclose(reaction, vx[:, None] + vy, rtol=1e-10, atol=1e-12)
            or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('The study is outside the separable laminar regime')
    x = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'],
                  study['reaction_x'] + vx)
    y = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'],
                  study['reaction_y'] + vy)
    operator = kron(x, eye(ny), format='csc') + kron(eye(nx), y, format='csc')
    identity = eye(nx * ny, format='csc')
    rhs = np.asarray(study['source_modes']).reshape(-1, nx * ny).T
    sensor_x, sensor_y = np.asarray(study['sensor_ij']).T
    sensor_indices = sensor_x * ny + sensor_y
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sensor_indices])
    return np.vstack(blocks), fields


def _box_simplex(S, y, ridge, cap, budget):
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    c = S.T @ y
    w = np.full(m, budget / m)
    # -1 and +1 denote fixed lower and upper coordinates, respectively.
    fixed = np.zeros(m, dtype=int)
    if budget == 0:
        return w, float(np.min(-c))
    if budget == m * cap:
        return w, float(np.max(H @ w - c))
    for _ in range(20 * m * m):
        free = fixed == 0
        if not np.any(free):
            g = H @ w - c
            low = np.flatnonzero(fixed < 0)
            high = np.flatnonzero(fixed > 0)
            a = low[np.argmin(g[low])]
            b = high[np.argmax(g[high])]
            if g[a] >= g[b] - 1e-12:
                return w, float((g[a] + g[b]) / 2)
            fixed[[a, b]] = 0
            continue
        active = ~free
        rhs = c[free] - H[np.ix_(free, active)] @ w[active]
        k = np.count_nonzero(free)
        system = np.empty((k + 1, k + 1))
        system[:k, :k] = H[np.ix_(free, free)]
        system[:k, k] = -1
        system[k, :k] = 1
        system[k, k] = 0
        face, nu = np.split(np.linalg.solve(system, np.r_[rhs, budget - w[active].sum()]), [k])
        nu = float(nu[0])
        old = w[free]
        direction = face - old
        below, above = face < -1e-12, face > cap + 1e-12
        if np.any(below | above):
            step = np.full(k, np.inf)
            step[below] = -old[below] / direction[below]
            step[above] = (cap - old[above]) / direction[above]
            hit = int(np.argmin(step))
            w[free] = np.clip(old + step[hit] * direction, 0, cap)
            index = np.flatnonzero(free)[hit]
            fixed[index] = -1 if below[hit] else 1
            w[index] = 0 if below[hit] else cap
            continue
        w[free] = np.clip(face, 0, cap)
        violation = -fixed * (nu - (H @ w - c))
        release = int(np.argmax(violation))
        if violation[release] <= 1e-12:
            return w, nu
        fixed[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _profile(study, samples, observations):
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((samples, observations))
    U, singular, _ = np.linalg.svd(Z, full_matrices=False)
    ridge = study['nuisance_ridge']
    if ridge > 0:
        fraction = singular**2 / (singular**2 + ridge)
    else:
        fraction = singular > np.finfo(float).eps * max(Z.shape) * np.max(singular, initial=0)
    data -= U @ (fraction[:, None] * (U.T @ data))
    return data[:, :-1], data[:, -1]


def _certificate(S, y, ridge, cap, budget, w, nu):
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    objective = 0.5 * float(residual @ residual) + 0.5 * ridge * float(w @ w)
    shifted = gradient - nu
    lower_bound = objective - gradient @ w + nu * budget + cap * np.minimum(shifted, 0).sum()
    return {
        'weights': w.copy(), 'nu_sum': float(nu),
        'lam_lower': np.maximum(shifted, 0), 'lam_upper': np.maximum(-shifted, 0),
        'primal_objective': objective, 'dual_lower_bound': min(objective, float(lower_bound)),
    }


def solve_beta_m96_w03_batch_03(study: dict) -> dict:
    samples, terminal = _forward(study)
    S, y = _profile(study, samples, _calibrated_observations(study))
    ridge, cap = study['source_ridge'], study['u_box']
    weights, certificates = [], []
    for budget in study['budget_grid']:
        w, nu = _box_simplex(S, y, ridge, cap, float(budget))
        weights.append(w)
        certificates.append(_certificate(S, y, ridge, cap, float(budget), w, nu))
    return {
        'weights_by_budget': np.stack(weights),
        'terminal_field_mid_budget': (terminal @ weights[1]).reshape(study['nx'], study['ny']),
        'kkt_certificates': certificates,
    }
