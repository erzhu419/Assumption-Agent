"""Recover source weights for a separable shifted transport model."""

import numpy as np
from scipy.linalg import eigh_tridiagonal, helmert


def _axis_spectrum(length, diffusivity, velocity, robin):
    """Diagonalize the cell-centered axis operator after diagonal similarity."""
    k = np.asarray(diffusivity, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    face = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    lower = face + np.maximum(v[1:], 0) / h
    upper = face + np.maximum(-v[:-1], 0) / h
    diag = np.abs(v) / h
    diag[:-1] += face
    diag[1:] += face
    for i in (0, k.size - 1):
        diag[i] += robin * k[i] / (h * (k[i] + 0.5 * robin * h))
    log_scale = np.r_[0.0, 0.5 * np.cumsum(np.log(lower / upper))]
    scale = np.exp(log_scale - (log_scale.max() + log_scale.min()) / 2)
    eig, q = eigh_tridiagonal(diag, -np.sqrt(lower * upper))
    return eig, scale[:, None] * q, q.T / scale


def _fit(design, target, cap, budget, ridge):
    """Solve the equality-constrained least squares on successive box faces."""
    m = design.shape[1]
    if budget == 0 or budget == cap * m:
        return np.full(m, budget / m)
    a = np.vstack((design, np.sqrt(ridge) * np.eye(m)))
    b = np.r_[target, np.zeros(m)]
    w = np.full(m, budget / m)
    state = np.zeros(m, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(20 * m * m):
        free = np.flatnonzero(state == 0)
        if free.size > 1:
            basis = helmert(free.size).T
            step = basis @ np.linalg.lstsq(a[:, free] @ basis, b - a @ w, rcond=None)[0]
            trial = w[free] + step
            hits = np.flatnonzero((trial < 0) | (trial > cap))
            if hits.size:
                bound = np.where(step[hits] > 0, cap, 0.0)
                frac = (bound - w[free[hits]]) / step[hits]
                first = frac.argmin()
                w[free] = np.clip(w[free] + frac[first] * step, 0.0, cap)
                idx = free[hits[first]]
                w[idx] = bound[first]
                state[idx] = 1 if bound[first] == cap else -1
                continue
            w[free] = trial
        g = a.T @ (a @ w - b)
        tol = 2e-13 * max(1.0, np.max(np.abs(g)))
        if free.size:
            level = g[free].mean()
            violation = np.where(state < 0, level - g, np.where(state > 0, g - level, 0.0))
            idx = violation.argmax()
            if violation[idx] > tol:
                state[idx] = 0
                continue
        else:
            low = np.flatnonzero(state < 0)
            high = np.flatnonzero(state > 0)
            receiver = low[g[low].argmin()]
            donor = high[g[high].argmax()]
            if g[donor] - g[receiver] > tol:
                state[[receiver, donor]] = 0
                continue
        defect = budget - w.sum()
        idx = np.argmax(cap - w if defect >= 0 else w)
        w[idx] = np.clip(w[idx] + defect, 0.0, cap)
        return w
    raise RuntimeError('Box-simplex active set did not converge')


def solve_kronecker_box_simplex_recovery(study: dict) -> dict:
    """Return feasible weights, their last-shift field, and a KKT certificate."""
    sources = np.asarray(study['source_modes'], dtype=float)
    shifts = np.asarray(study['mu_shifts'], dtype=float)
    (ex, vx, ix), (ey, vy, iy) = [
        _axis_spectrum(study['L' + d], study['kappa_' + d],
                       study['vel_' + d], study['robin_' + d]) for d in 'xy'
    ]
    rhs = ix @ sources @ iy.T
    denom = ex[:, None] + ey[None, :] + shifts[:, None, None]
    fields = vx @ (rhs[None, ...] / denom[:, None, ...]) @ vy.T
    i, j = np.asarray(study['sensor_ij']).T
    sampled = fields[:, :, i, j].transpose(0, 2, 1).reshape(-1, sources.shape[0])
    z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((sampled, study['y_obs']))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    profiled = values - z @ np.linalg.solve(gram, z.T @ values)
    design, target = profiled[:, :-1], profiled[:, -1]
    cap, budget, ridge = (float(study[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    w = _fit(design, target, cap, budget, ridge)
    res = design @ w - target
    primal = float(0.5 * (res @ res) + 0.5 * ridge * (w @ w))
    g = design.T @ res + ridge * w
    # An order statistic maximizes the box-Lagrange supporting-plane bound.
    rank = min(int(budget / cap), w.size - 1) if cap else 0
    nu = float(np.partition(g, rank)[rank])
    lower = np.maximum(g - nu, 0.0)
    upper = np.maximum(nu - g, 0.0)
    gap = float(lower @ w + upper @ (cap - w) + nu * (w.sum() - budget))
    cert = dict(weights=w.tolist(), nu_sum=nu, lam_lower=lower.tolist(),
                lam_upper=upper.tolist(), primal_objective=primal,
                dual_lower_bound=primal - max(0.0, gap))
    return dict(weights=w, terminal_field=np.einsum('m,mij->ij', w, fields[-1]),
                kkt_certificate=cert)
