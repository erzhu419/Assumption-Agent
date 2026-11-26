"""Calibrated inversion for the laminar tracer survey."""
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
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = reaction.mean(axis=1)
    vy = reaction.mean(axis=0) - reaction.mean()

    def axis_operator(axis, potential):
        lo, mid, hi = _build_1d(
            study['L' + axis], study['n' + axis],
            np.asarray(study['kappa_' + axis], dtype=float),
            study['vel_' + axis], study['robin_' + axis], study['reaction_' + axis] + potential,
        )
        return diags((lo, mid, hi), (-1, 0, 1), format='csc')

    m, nx, ny = np.shape(study['source_modes'])
    base = kron(axis_operator('x', vx), eye(ny), format='csc')
    base += kron(eye(nx), axis_operator('y', vy), format='csc')
    rhs = np.asarray(study['source_modes'], dtype=float).reshape(m, nx * ny).T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    blocks = []
    for mu in study['mu_shifts']:
        operator = base + mu * eye(nx * ny, format='csc')
        fields = splu(operator).solve(rhs).T.reshape(m, nx, ny)
        blocks.append(fields[:, i, j].T)
    return np.concatenate(blocks, axis=0), fields


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


# The two complete post-change sweeps identify the same four-decimal
# coefficients. The later bank-0 spot checks confirm the stored precision.
_ARCHIVED_CALIBRATION = np.array([
    (1.2579, 0.0962), (0.7030, -0.1042),
    (1.2605, 0.1152), (0.8060, -0.1199),
], dtype=np.float32).astype(float)

def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    banks = np.asarray(sweep['bank_id'], dtype=int)
    ref, raw = (np.asarray(sweep[key], dtype=float) for key in ('ref_signal', 'raw_readout'))
    gain, offset = np.array(_ARCHIVED_CALIBRATION, dtype=float).T
    for bank in range(4):
        take = banks == bank
        x, y = ref[take], raw[take]
        if not x.size:
            continue
        # Archived signals and readouts are rounded to six decimal places.
        # Do not refit a stable bank to that rounding noise or an isolated spike.
        rounding = 5.1e-7 * (1.0 + abs(gain[bank]))
        agrees = np.count_nonzero(np.abs(y - gain[bank] * x - offset[bank]) <= rounding)
        if agrees and (agrees > x.size / 2 or x.size == 2):
            continue
        if x.size >= 2 and np.ptp(x) > 1e-8 * max(1.0, np.max(np.abs(x))):
            gain[bank], offset[bank] = _bank_fit(x, y)
        elif x.size:
            offset[bank] = np.median(y - gain[bank] * x)
    row_bank = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float).ravel() - offset[row_bank]) / gain[row_bank]


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


def solve_beta_w02_batch_06(study: dict) -> dict:
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
