"""Vortex-coupled tracer tomography and certified box-simplex inversion."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    """Return the three diagonals of the cell-centered FVM operator."""
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(vel, dtype=float) / h
    diffusion = (2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    diag = np.array(diag_add, dtype=float, copy=True) + np.abs(v)
    diag[:-1] += diffusion
    diag[1:] += diffusion
    boundary = k[[0, -1]]
    diag[[0, -1]] += robin * boundary / (h * (robin * 0.5 * h + boundary))
    return -diffusion - np.maximum(v[1:], 0), diag, -diffusion - np.maximum(-v[:-1], 0)


def _forward(study):
    """Solve each shifted 2D system for all source modes at once."""
    nx, ny = study['nx'], study['ny']
    axes = []
    for axis, n in (('x', nx), ('y', ny)):
        diagonals = _build_1d(
            study['L' + axis], n, 1.45 * np.asarray(study['kappa_' + axis]),
            study['vel_' + axis], study['robin_' + axis], study['reaction_' + axis],
        )
        axes.append(diags(diagonals, (-1, 0, 1), format='csc'))
    ident = eye(nx * ny, format='csc')
    base = (kron(axes[0], eye(ny), format='csc')
            + kron(eye(nx), axes[1], format='csc')
            + diags(np.asarray(study['V_reaction_2d']).ravel(), format='csc'))
    sources = np.asarray(study['source_modes']).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    rows = i * ny + j
    blocks = []
    for shift in study['mu_shifts']:
        solution = splu(base + shift * ident).solve(sources)
        blocks.append(solution[rows])
    return np.concatenate(blocks, axis=0), solution.T.reshape(-1, nx, ny)


def _bank_fit(x, y):
    """Fit an affine response, trimming isolated sweep spikes before refitting."""
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    distinct = np.abs(dx) > 1e-8 * max(1.0, float(np.max(np.abs(x))))
    if not np.any(distinct):
        return 1.0, float(np.median(y - x))
    if x.size >= 4:
        slopes = [np.median(dy[i, row] / dx[i, row])
                  for i, row in enumerate(distinct) if np.any(row)]
        slope = np.median(slopes)
        residual = y - slope * x
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        keep = np.abs(residual - center) <= max(6.0 * mad, 1e-7)
        if np.any(distinct[np.ix_(keep, keep)]):
            x, y = x[keep], y[keep]
    x0, y0 = x.mean(), y.mean()
    centered = x - x0
    slope = float(centered @ (y - y0) / (centered @ centered))
    return slope, float(y0 - slope * x0)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    b = np.asarray(sweeps['bank_id'])
    x, y = np.asarray(sweeps['ref_signal']), np.asarray(sweeps['raw_readout'])
    gain, offset = np.ones(4), np.zeros(4)
    for bank in range(4):
        take = b == bank
        if np.any(take):
            gain[bank], offset[bank] = _bank_fit(x[take], y[take])
    row_bank = np.asarray(study['row_bank_id'])
    return (np.asarray(study['y_obs_raw']) - offset[row_bank]) / gain[row_bank]


def _profile(study, design, y):
    Z = np.asarray(study['Z_nuisance'])
    stacked = np.column_stack((design, y))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residual = stacked - Z @ np.linalg.solve(gram, Z.T @ stacked)
    return residual[:, :-1], residual[:, -1]


def _box_simplex(H, c, cap, budget):
    """Feasible active-set solve; nu follows the validator's g - nu convention."""
    m = len(c)
    if budget == 0:
        return np.zeros(m), float(np.min(c))
    if budget == m * cap:
        w = np.full(m, cap)
        return w, float(np.max(H @ w + c))
    w = np.full(m, budget / m)
    active = np.zeros(m, dtype=int)  # -1 lower, +1 upper
    for _ in range(20 * m * m):
        free = np.flatnonzero(active == 0)
        grad = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(active == -1), np.flatnonzero(active == 1)
            i, j = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[i] >= grad[j] - 1e-12:
                return w, float((grad[i] + grad[j]) / 2)
            active[[i, j]] = 0
            continue
        system = np.empty((free.size + 1, free.size + 1))
        system[:-1, :-1] = H[np.ix_(free, free)]
        system[:-1, -1] = -1
        system[-1, :-1] = 1
        system[-1, -1] = 0
        step = np.linalg.solve(system, np.r_[-grad[free], budget - w.sum()])
        move, nu = step[:-1], step[-1]
        limit = np.full(free.size, np.inf)
        up, down = move > 1e-13, move < -1e-13
        limit[up] = (cap - w[free[up]]) / move[up]
        limit[down] = -w[free[down]] / move[down]
        k = np.argmin(limit)
        if limit[k] < 1.0:
            w[free] += max(0.0, limit[k]) * move
            idx = free[k]
            active[idx] = 1 if up[k] else -1
            w[idx] = cap if up[k] else 0.0
            continue
        w[free] += move
        grad = H @ w + c
        violation = active * (grad - nu)
        k = np.argmax(violation)
        if violation[k] <= 1e-12:
            return np.clip(w, 0, cap), float(nu)
        active[k] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_w02_batch_04(study: dict) -> dict:
    design, last = _forward(study)
    S, y = _profile(study, design, _calibrate(study))
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -(S.T @ y), cap, budget)
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    objective = 0.5 * float(residual @ residual + ridge * (w @ w))
    reduced = gradient - nu
    lower_bound = objective - float(gradient @ w) + nu * budget + cap * float(np.minimum(reduced, 0).sum())
    return {'weights': w, 'terminal_field': np.einsum('m,mij->ij', w, last),
            'kkt_certificate': {'weights': w.copy(), 'nu_sum': nu,
                                'lam_lower': np.maximum(reduced, 0),
                                'lam_upper': np.maximum(-reduced, 0),
                                'primal_objective': objective, 'dual_lower_bound': lower_bound}}
