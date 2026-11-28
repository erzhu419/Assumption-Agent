"""Calibrated inversion for the periodic-x, closed-y transport chamber."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import root


def _sensor_features(x, k):
    v = x[..., k]
    prev = x[..., (k - 1) % 4]
    nxt = x[..., (k + 1) % 4]
    return np.stack((np.ones_like(v), v, prev, nxt, v * nxt, v**3), axis=-1)


def _sensor_coefficients(sweeps):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    bank_ref = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
    bank_raw = np.asarray(bank['raw_readout'])
    # Discard gross electronic faults, separated from the ordinary sweep range.
    good = np.max(np.abs(raw), axis=1) < 8
    coefficients = []
    for k in range(4):
        use = (ids == k) & (np.abs(bank_raw) < 8)
        x = np.concatenate((ref[good], bank_ref[use]))
        y = np.concatenate((raw[good, k], bank_raw[use]))
        coefficients.append(np.linalg.lstsq(_sensor_features(x, k), y, rcond=None)[0])
    return np.asarray(coefficients)


def _calibrate(study):
    c = _sensor_coefficients(study['sensor_calibration_sweeps'])
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    linear = np.diag(c[:, 1])
    k = np.arange(4)
    linear[k, (k - 1) % 4] = c[:, 2]
    linear[k, (k + 1) % 4] = c[:, 3]
    initial = np.linalg.solve(linear, (observed - c[:, 0]).T).T

    def response(x):
        return np.array([_sensor_features(x, j) @ c[j] for j in range(4)])

    calibrated = np.empty_like(initial)
    for j, (y, x0) in enumerate(zip(observed, initial)):
        sol = root(lambda x: response(x) - y, x0, tol=1e-11)
        if np.linalg.norm(sol.fun, np.inf) > 1e-9:
            raise RuntimeError('rosette inversion did not converge')
        calibrated[j] = sol.x
    return calibrated.ravel()


def _axis(kappa, vel, h, periodic):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    n = len(kappa)
    face = (kappa[1:] + kappa[:-1]) / (2 * h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(vel[1:-1], 0) / h
    upper[1:] -= np.maximum(-vel[1:-1], 0) / h
    diagonal = np.empty(n)
    diagonal[1:-1] = -lower[:-1] - upper[1:]
    if periodic:
        seam = (kappa[0] + kappa[-1]) / (2 * h**2)
        upper[0] -= max(-vel[0], 0) / h
        lower[-1] -= max(vel[-1], 0) / h
        wrap_lo = -seam - max(vel[0], 0) / h
        wrap_hi = -seam - max(-vel[-1], 0) / h
        diagonal[0] = -upper[0] - wrap_lo
        diagonal[-1] = -lower[-1] - wrap_hi
        return sparse.diags(([wrap_hi], lower, diagonal, upper, [wrap_lo]),
                            (1 - n, -1, 0, 1, n - 1), format='csc')
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    diagonal[0], diagonal[-1] = -upper[0], -lower[-1]
    return sparse.diags((lower, diagonal, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), True)
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), False)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    rhs = np.column_stack([np.asarray(mode).ravel(order='F') for mode in study['source_modes']])
    ij = np.asarray(study['sensor_ij'])
    rows = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    sampled = []
    for shift in study['mu_shifts']:
        sol = splu(operator + shift * eye).solve(rhs)
        sampled.append(sol[rows])
    return np.concatenate(sampled), sol.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _box_simplex(H, b, cap, budget):
    """Primal active-set QP; the gradient is H @ w - b."""
    n = len(b)
    if budget == 0:
        w = np.zeros(n)
        return w, float(np.min(-b))
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float(np.max(H @ w - b))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    tol = 64 * np.finfo(float).eps * max(1., np.max(np.abs(b)), cap * np.linalg.norm(H, np.inf))
    for _ in range(20 * n * n):
        free = active == 0
        lower, upper = active < 0, active > 0
        if not np.any(free):
            g = H @ w - b
            lo = np.argmin(np.where(lower, g, np.inf))
            hi = np.argmax(np.where(upper, g, -np.inf))
            if g[hi] <= g[lo] + tol:
                return w, float((g[hi] + g[lo]) / 2)
            active[[lo, hi]] = 0
            continue
        count = np.count_nonzero(free)
        system = np.zeros((count + 1, count + 1))
        system[:-1, :-1] = H[np.ix_(free, free)]
        system[:-1, -1] = -1
        system[-1, :-1] = 1
        rhs = np.r_[b[free] - H[np.ix_(free, upper)] @ w[upper], budget - np.sum(w[upper])]
        result = np.linalg.solve(system, rhs)
        target = result[:-1]
        nu = float(result[-1])
        idx = np.flatnonzero(free)
        cur = w[idx]
        delta = target - cur
        steps = np.full(count, np.inf)
        below = (target < 0) & (delta < 0)
        above = (target > cap) & (delta > 0)
        steps[below] = -cur[below] / delta[below]
        steps[above] = (cap - cur[above]) / delta[above]
        hit = np.argmin(steps)
        if steps[hit] < 1:
            w[idx] = np.clip(cur + steps[hit] * delta, 0, cap)
            active[idx[hit]] = -1 if below[hit] else 1
            w[idx[hit]] = 0 if below[hit] else cap
            continue
        w[idx] = np.clip(target, 0, cap)
        g = H @ w - b
        violation = np.where(lower, nu - g, np.where(upper, g - nu, -np.inf))
        release = np.argmax(violation)
        if violation[release] <= tol:
            return w, nu
        active[release] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    y = _calibrate(study)
    S, terminal_modes = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    profile = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    design = S - Z @ (profile @ S)
    data = y - Z @ (profile @ y)
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    budget = float(study['total_budget'])
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, design.T @ data, cap, budget)
    residual = design @ w - data
    g = design.T @ residual + ridge * w
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    lower = np.maximum(0, g - nu)
    upper = np.maximum(0, nu - g)
    dual = float(objective - g @ w + nu * budget - cap * np.sum(upper))
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
            'primal_objective': objective, 'dual_lower_bound': dual,
        },
    }
