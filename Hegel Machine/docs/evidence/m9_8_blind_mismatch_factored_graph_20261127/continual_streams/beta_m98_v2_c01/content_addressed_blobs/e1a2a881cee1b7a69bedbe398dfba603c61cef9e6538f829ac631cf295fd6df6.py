"""Periodic transport tomography with sweep-fitted rosettes and a certified box-simplex solution."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis(kappa, velocity, h):
    """Periodic diffusion and row-local upwind, including the wrap face."""
    face = (kappa + np.roll(kappa, -1)) * (0.5 / h**2)
    left = np.roll(face, 1) + np.maximum(velocity, 0) / h
    right = face + np.maximum(-velocity, 0) / h
    axis = diags((-left[1:], left + right, -right[:-1]), (-1, 0, 1), format='lil')
    axis[0, -1] += -left[0]
    axis[-1, 0] += -right[-1]
    return axis.tocsc()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1))
    operator = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
                + diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.stack([mode.ravel(order='F') for mode in study['source_modes']], axis=1)
    ij = np.asarray(study['sensor_ij'])
    sensor_rows = ij[:, 0] + nx * ij[:, 1]
    identity = eye(nx * ny, format='csc')
    observations = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(rhs)
        observations.append(fields[sensor_rows])
    terminal_modes = np.stack([field.reshape((nx, ny), order='F') for field in fields.T])
    return np.vstack(observations), terminal_modes


def _calibration_features(x, j):
    prev, nxt = (j - 1) % 4, (j + 1) % 4
    u = x[:, j]
    return np.column_stack((np.ones(len(x)), x[:, prev], u, x[:, nxt], u * x[:, nxt], u**3))


def _fit_calibration(ref, raw):
    return np.stack([np.linalg.lstsq(_calibration_features(ref, j), raw[:, j], rcond=None)[0]
                     for j in range(4)])


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    # Omit the gross readout faults, separated from the ordinary sweeps.
    good = np.max(np.abs(raw), axis=1) < 9
    a = _fit_calibration(ref[good], raw[good])
    obs = study['y_obs_raw'].reshape(-1, 4)
    J = np.zeros((len(obs), 4, 4))
    for j in range(4):
        J[:, j, [(j-1)%4, j, (j+1)%4]] = a[j, 1:4]
    x = np.linalg.solve(J[0], (obs - a[:, 0]).T).T
    for _ in range(12):
        f = np.column_stack([_calibration_features(x, j) @ a[j] for j in range(4)]) - obs
        if np.max(np.abs(f)) < 1e-13:
            break
        for j in range(4):
            nxt = (j+1)%4
            J[:, j, j] = a[j, 2] + a[j, 4]*x[:, nxt] + 3*a[j, 5]*x[:, j]**2
            J[:, j, nxt] = a[j, 3] + a[j, 4]*x[:, j]
        x -= np.linalg.solve(J, f[..., None])[..., 0]
    return x.ravel()


def _box_simplex_qp(hessian, linear, cap, budget):
    """Feasible active set; -1 and +1 mark lower and upper bounds."""
    n = len(linear)
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=np.int8)
    if budget == 0:
        return np.zeros(n), float(linear.min())
    if budget == n * cap:
        w.fill(cap)
        return w, float((hessian @ w + linear).max())
    tol = 1e-11 * max(1.0, np.max(np.abs(linear)), cap * np.max(np.abs(hessian)))
    for _ in range(20 * n + 100):
        g = hessian @ w + linear
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active == -1)
        upper = np.flatnonzero(active == 1)
        if not free.size:
            lo = lower[np.argmin(g[lower])]
            hi = upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue

        h = hessian[np.ix_(free, free)]
        system = np.ones((len(free) + 1, len(free) + 1))
        system[:-1, :-1] = h
        system[:-1, -1] = -1
        system[-1, -1] = 0
        direction_nu = np.linalg.solve(system, np.r_[-g[free], budget - w.sum()])
        direction, nu = direction_nu[:-1], float(direction_nu[-1])
        # Only truncate when the full equality-constrained step crosses a bound.
        limits = np.full(len(free), np.inf)
        down = (direction < 0) & (w[free] + direction < -1e-14 * cap)
        up = (direction > 0) & (w[free] + direction > cap + 1e-14 * cap)
        limits[down] = -w[free[down]] / direction[down]
        limits[up] = (cap - w[free[up]]) / direction[up]
        hit = int(np.argmin(limits))
        if limits[hit] < 1:
            w[free] += max(0., limits[hit]) * direction
            idx = free[hit]
            active[idx] = 1 if up[hit] else -1
            w[idx] = cap if up[hit] else 0.
            continue

        w[free] = np.clip(w[free] + direction, 0, cap)
        g = hessian @ w + linear
        violation = active * (g - nu)
        release = int(np.argmax(violation))
        if violation[release] <= tol:
            return w, nu
        active[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    design, terminal_modes = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    data = np.column_stack((design, y))
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    s, y = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    w, nu = _box_simplex_qp(s.T @ s + ridge * np.eye(s.shape[1]), -s.T @ y, cap, budget)
    residual = s @ w - y
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = s.T @ residual + ridge * w
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective,
            'dual_lower_bound': float(objective - g @ w + nu * budget - cap * upper.sum()),
        },
    }
