"""Shifted separable transport and certified box-simplex inversion."""

import numpy as np


def _axis_spectrum(length, kappa, velocity, robin):
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(velocity, dtype=float)
    h = length / k.size
    diffusion = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    upper = diffusion + np.maximum(-v[:-1], 0) / h
    lower = diffusion + np.maximum(v[1:], 0) / h
    diagonal = np.zeros_like(k)
    diagonal[:-1] += upper
    diagonal[1:] += lower
    diagonal[[0, -1]] += (robin / h) / (1 + robin * h / (2 * k[[0, -1]]))
    diagonal[0] += max(v[0], 0) / h
    diagonal[-1] += max(-v[-1], 0) / h

    # A = D T D^-1, with T real symmetric tridiagonal.
    log_scale = np.r_[0., .5 * np.cumsum(np.log(lower / upper))]
    scale = np.exp(log_scale - log_scale.mean())
    off = -np.sqrt(lower * upper)
    eig, q = np.linalg.eigh(np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1))
    return eig, scale[:, None] * q, q.T / scale


def _transport(study):
    ex, vx, ix = _axis_spectrum(study['Lx'], study['kappa_x'], study['vel_x'], study['robin_x'])
    ey, vy, iy = _axis_spectrum(study['Ly'], study['kappa_y'], study['vel_y'], study['robin_y'])
    source = np.asarray(study['source_modes'], dtype=float)
    si, sj = np.asarray(study['sensor_ij'], dtype=int).T
    transformed = ix @ source @ iy.T
    axis_sum = ex[:, None] + ey
    sensors = []
    for mu in study['mu_shifts']:
        spectral_field = transformed / (axis_sum + mu)
        # Only reconstruct the required sensor samples until the terminal shift.
        sensors.append(np.einsum('si,mis->sm', vx[si], spectral_field @ vy[sj].T))
    terminal = vx @ spectral_field @ vy.T
    return np.concatenate(sensors, axis=0), terminal


def _profile(design, values, nuisance, ridge):
    joined = np.column_stack((design, values))
    z = np.asarray(nuisance, dtype=float)
    if z.shape[1]:
        if ridge == 0:
            coef = np.linalg.lstsq(z, joined, rcond=None)[0]
        else:
            coef = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ joined)
        joined -= z @ coef
    return joined[:, :-1], joined[:, -1]


def _box_simplex(hessian, linear, cap, budget):
    """Equality-constrained active set, releasing bounds by reduced gradient."""
    n = linear.size
    capacity = n * cap
    if budget < 0 or budget > capacity:
        raise ValueError('Budget is outside box-simplex capacity')
    if budget == 0:
        return np.zeros(n)
    if budget == capacity:
        return np.full(n, cap)

    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    # Require both small reduced gradients and a gap well within 1e-6.
    tol = min(1e-10, 1e-8 / max(1., capacity))
    for _ in range(100 * n):
        g = hessian @ w + linear
        free = np.flatnonzero(status == 0)
        if free.size == 0:
            up = np.flatnonzero(status == 1)
            lo = np.flatnonzero(status == -1)
            donor = up[np.argmax(g[up])]
            receiver = lo[np.argmin(g[lo])]
            if g[donor] - g[receiver] <= tol:
                return w
            status[[donor, receiver]] = 0
            continue

        f = free.size
        system = np.zeros((f + 1, f + 1))
        system[:f, :f] = hessian[np.ix_(free, free)]
        system[:f, f] = system[f, :f] = 1.
        rhs = np.r_[-g[free], 0.]
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        d = solution[:f]
        d -= d.mean()
        stationarity = g[free] + system[:f, :f] @ d
        stationarity -= stationarity.mean()
        if np.max(np.abs(stationarity)) > tol and np.max(np.abs(stationarity)) > 1e-12 * np.max(np.abs(g[free])):
            # A singular face can have a linear descent direction in its null space.
            d = -stationarity / np.max(np.abs(stationarity)) * (2 * cap)
        wf = w[free]
        hit = np.full(f, np.inf)
        lower = (d < 0) & (wf + d < 0)
        upper = (d > 0) & (wf + d > cap)
        hit[lower] = -wf[lower] / d[lower]
        hit[upper] = (cap - wf[upper]) / d[upper]
        j = int(np.argmin(hit))
        if hit[j] < 1.:
            w[free] = np.clip(wf + hit[j] * d, 0., cap)
            w[free[j]] = cap if upper[j] else 0.
            status[free[j]] = 1 if upper[j] else -1
            continue
        w[free] = np.clip(wf + d, 0., cap)
        # Repair equality roundoff on the free coordinate with most room.
        err = budget - w.sum()
        room = cap - w[free] if err > 0 else w[free]
        j = free[np.argmax(room)]
        w[j] = np.clip(w[j] + err, 0., cap)
        g = hessian @ w + linear
        nu = g[free].mean()
        violation = np.where(status == -1, nu - g, np.where(status == 1, g - nu, 0.))
        j = int(np.argmax(violation))
        if violation[j] <= tol:
            return w
        status[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def _certificate(design, values, ridge, cap, budget, w):
    residual = design @ w - values
    g = design.T @ residual + ridge * w
    donors = w > 0
    receivers = w < cap
    if not donors.any():
        nu = g.min()
    elif not receivers.any():
        nu = g.max()
    else:
        nu = .5 * (g[donors].max() + g[receivers].min())
    lower = np.maximum(g - nu, 0.)
    upper = np.maximum(nu - g, 0.)
    primal = .5 * (residual @ residual + ridge * (w @ w))
    # Algebraically J - L_KKT, without subtracting two nearly equal objectives.
    gap = lower @ w + upper @ (cap - w) + nu * (w.sum() - budget)
    return {'weights': w.tolist(), 'nu_sum': float(nu),
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': float(primal),
            'dual_lower_bound': float(primal - max(0., gap))}


def solve_multibudget_kronecker_transfer(study: dict) -> dict:
    stacked, terminal = _transport(study)
    design, values = _profile(stacked, study['y_obs'], study['Z_nuisance'], study['nuisance_ridge'])
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    linear = -design.T @ values
    budgets = [float(b) for b in study['budget_grid']]
    weights = np.stack([_box_simplex(hessian, linear, cap, b) for b in budgets])
    certs = [_certificate(design, values, ridge, cap, b, w) for b, w in zip(budgets, weights)]
    return {'weights_by_budget': weights,
            'terminal_field_mid_budget': np.einsum('m,mij->ij', weights[1], terminal),
            'kkt_certificates': certs}
