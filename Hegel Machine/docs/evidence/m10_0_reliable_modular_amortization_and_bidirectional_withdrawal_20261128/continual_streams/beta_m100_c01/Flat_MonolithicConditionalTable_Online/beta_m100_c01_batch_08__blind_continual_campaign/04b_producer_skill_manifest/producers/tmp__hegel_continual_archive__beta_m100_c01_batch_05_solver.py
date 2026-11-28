import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _response(x, p):
    bias, gain = p[:4], p[4:8]
    amp, freq, cross, even, cross_freq = p[8:]
    opposite = np.roll(x, 2, axis=1)
    return bias + gain * (x + amp*np.sin(freq*x) + cross*opposite
                          + even*(np.cos(cross_freq*opposite) - 1))


def _fit_calibration(x, y, banks):
    # Large injected offsets in the sweeps are separated from the signal range.
    clean = np.max(np.abs(y), axis=1) < 5
    single = np.abs(banks['raw_readout']) < 5
    idx, q, raw = (banks[k][single] for k in ('bank_id', 'ref_signal', 'raw_readout'))

    def residual(p):
        amp, freq = p[8:10]
        own = p[idx] + p[4 + idx]*(q + amp*np.sin(freq*q))
        return np.r_[(_response(x[clean], p) - y[clean]).ravel(), own - raw]

    initial = np.r_[np.zeros(4), np.ones(4), .2, 2.4, .2, .05, 2.2]
    return least_squares(residual, initial, ftol=1e-12, xtol=1e-12, gtol=1e-12).x


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    p = _fit_calibration(sweeps['ref_rosette'], sweeps['raw_rosette'], sweeps['single_bank_sweeps'])
    amp, freq, cross, even, cross_freq = p[8:]
    target = (np.asarray(study['y_obs_raw']).reshape(-1, 4) - p[:4]) / p[4:8]
    idx = np.arange(4)
    opp = (idx+2) % 4
    x = target / (1 + amp*freq)
    for _ in range(30):
        other = x[:, opp]
        residual = x + amp*np.sin(freq*x) + cross*other + even*(np.cos(cross_freq*other)-1) - target
        if np.max(np.abs(residual)) < 1e-13:
            return x.ravel()
        jac = np.zeros((len(x), 4, 4))
        jac[:, idx, idx] = 1 + amp*freq*np.cos(freq*x)
        jac[:, idx, opp] = cross - even*cross_freq*np.sin(cross_freq*other)
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise RuntimeError('Rosette inversion did not converge')


def _axis(kappa, vel, h, robin, mode):
    n = len(kappa)
    row = np.arange(n)
    left, right = (row - 1) % n, (row + 1) % n
    west = (kappa + kappa[left]) * (0.5 / h**2) + np.maximum(vel, 0) / h
    east = (kappa + kappa[right]) * (0.5 / h**2) + np.maximum(-vel, 0) / h
    leak = np.zeros(n)
    if mode != 'P':
        west[0] = east[-1] = 0
        east[0], west[-1] = kappa[0] / h**2, kappa[-1] / h**2
        if mode == 'R':
            leak[[0, -1]] = robin / h
    return csc_matrix((np.r_[-west, west + east + leak, -east],
                       (np.tile(row, 3), np.r_[left, row, right])), shape=(n, n))


def _transport(study, walls='RR'):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a], study['L' + a] / (n - 1),
                    study['robin_' + a], mode)
              for a, n, mode in zip('xy', (nx, ny), walls)]
    reaction = diags(study['V_reaction_2d'].ravel(order='F'), format='csc')
    op = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
    identity = eye(nx * ny, format='csc')
    modes = study['source_modes']
    rhs = np.column_stack([mode.ravel(order='F') for mode in modes])
    ij = np.asarray(study['sensor_ij'], dtype=int)
    slots = ij[:, 0] + nx * ij[:, 1]
    blocks = []
    for shift in study['mu_shifts']:
        solution = splu(op + shift*identity).solve(rhs)
        blocks.append(solution[slots])
    return np.vstack(blocks), np.stack([col.reshape((nx, ny), order='F') for col in solution.T])


def _box_simplex(H, c, cap, budget):
    """Feasible active-set QP; nu uses the sign convention g - nu."""
    n = len(c)
    if budget == 0 or budget == n * cap:
        w = np.full(n, 0.0 if budget == 0 else cap)
        g = H @ w + c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    tol = 1e-12 * max(1.0, np.max(np.abs(H)) * cap, np.max(np.abs(c)))
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        g = H @ w + c
        if not free.size:
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + tol:
                return w, float((g[j] + g[i]) / 2)
            state[[i, j]] = 0
            continue
        F = H[np.ix_(free, free)]
        # Solve for the face minimizer, enforcing the remaining sum exactly.
        rhs = -c[free] - H[free] @ np.where(state == 0, 0.0, w)
        ones = np.ones(len(free))
        a, b = np.linalg.solve(F, np.column_stack((rhs, ones))).T
        nu = (budget - np.sum(w[state != 0]) - a.sum()) / b.sum()
        wf = a + nu * b
        step = wf - w[free]
        hits = np.full(len(free), np.inf)
        lower, upper = wf < -tol, wf > cap + tol
        hits[lower] = -w[free][lower] / step[lower]
        hits[upper] = (cap - w[free][upper]) / step[upper]
        k = int(np.argmin(hits))
        if hits[k] <= 1.0:
            w[free] += np.clip(hits[k], 0.0, 1.0) * step
            state[free[k]] = -1 if lower[k] else 1
            w[free[k]] = 0.0 if lower[k] else cap
            continue
        w[free] = np.clip(wf, 0.0, cap)
        g = H @ w + c
        violation = state * (g - nu)
        i = int(np.argmax(violation))
        if violation[i] <= tol:
            return w, float(nu)
        state[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _transport(study)
    y = _calibrate(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A = S - Z @ (Pz @ S)
    r = y - Z @ (Pz @ y)
    ridge = study['source_ridge']
    H = A.T @ A + ridge * np.eye(A.shape[1])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    w, nu = _box_simplex(H, -A.T @ r, cap, budget)
    residual = A @ w - r
    g = A.T @ residual + ridge * w
    J = float(0.5 * (residual @ residual + ridge * (w @ w)))
    reduced = g - nu
    return {
        'weights': w,
        'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
        'kkt_certificate': {
            'weights': w.tolist(), 'nu_sum': nu,
            'lam_lower': np.maximum(0.0, reduced),
            'lam_upper': np.maximum(0.0, -reduced),
            'primal_objective': J,
            'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0.0, reduced).sum()),
        },
    }
