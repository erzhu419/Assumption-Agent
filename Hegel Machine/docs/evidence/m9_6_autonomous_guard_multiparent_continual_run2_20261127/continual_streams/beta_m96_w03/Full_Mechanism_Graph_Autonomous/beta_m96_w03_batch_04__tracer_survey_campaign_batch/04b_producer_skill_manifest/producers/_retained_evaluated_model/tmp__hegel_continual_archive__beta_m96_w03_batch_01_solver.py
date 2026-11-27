"""Calibrated laminar tracer inversion with a box-simplex certificate."""
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


def _calibrated_observations(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    reference = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    rows = np.asarray(study['row_bank_id'])
    result = np.empty(rows.shape, dtype=float)
    for bank in np.unique(rows):
        x, y = reference[banks == bank], raw[banks == bank]
        i, j = np.triu_indices(x.size, 1)
        dx = x[j] - x[i]
        distinct = dx != 0
        if not np.any(distinct):
            raise ValueError(f'Bank {bank} needs calibration at distinct reference levels')
        slope = np.median((y[j[distinct]] - y[i[distinct]]) / dx[distinct])
        residual = y - slope * x
        residual -= np.median(residual)
        floor = 64 * np.finfo(float).eps * max(1., np.max(np.abs(y)))
        good = np.abs(residual) <= max(floor, 6 * np.median(np.abs(residual)))
        if np.ptp(x[good]) == 0:
            good = np.ones(x.size, dtype=bool)
        x, y = x[good], y[good]
        centered = x - x.mean()
        gain = (centered @ (y - y.mean())) / (centered @ centered)
        offset = y.mean() - gain * x.mean()
        selected = rows == bank
        result[selected] = (np.asarray(study['y_obs_raw'])[selected] - offset) / gain
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


def solve_beta_m96_w03_s00(study: dict) -> dict:
    samples, terminal = _forward(study)
    observations = _calibrated_observations(study)
    Z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((samples, observations))
    # Apply the same ridge-residual operator to the design and observations.
    U, singular, _ = np.linalg.svd(Z, full_matrices=False)
    ridge_z = study['nuisance_ridge']
    if ridge_z > 0:
        fraction = singular**2 / (singular**2 + ridge_z)
    else:
        fraction = singular > np.finfo(float).eps * max(Z.shape) * np.max(singular, initial=0)
    profiled = data - U @ (fraction[:, None] * (U.T @ data))
    S, y = profiled[:, :-1], profiled[:, -1]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, y, ridge, cap, budget)
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    objective = float((residual @ residual + ridge * (w @ w)) / 2)
    shifted = gradient - nu
    # Minimize the objective's affine supporting plane over the box at fixed sum.
    lower_bound = objective - gradient @ w + nu * budget + cap * np.minimum(shifted, 0).sum()
    certificate = {
        'weights': w.copy(), 'nu_sum': nu,
        'lam_lower': np.maximum(shifted, 0), 'lam_upper': np.maximum(-shifted, 0),
        'primal_objective': objective, 'dual_lower_bound': min(objective, float(lower_bound)),
    }
    return {'weights': w, 'terminal_field': (terminal @ w).reshape(study['nx'], study['ny']),
            'kkt_certificate': certificate}
