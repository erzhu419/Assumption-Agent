"""Steady transport and certified box-simplex source reconstruction."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _sensor_features(x, channel):
    v = x[..., channel]
    prev = x[..., (channel - 1) % 4]
    nxt = x[..., (channel + 1) % 4]
    return np.stack((np.ones_like(v), v, prev, nxt, v * nxt, v**3), axis=-1)


def _sensor_fit(sweeps):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    bank_ref = np.eye(4)[ids] * np.asarray(bank['ref_signal'])[:, None]
    bank_raw = np.asarray(bank['raw_readout'])
    # The observed electronic faults are far outside the calibration range.
    good = np.max(np.abs(raw), axis=1) < 8
    coeff = []
    for k in range(4):
        use = (ids == k) & (np.abs(bank_raw) < 8)
        x = np.concatenate((ref[good], bank_ref[use]))
        y = np.concatenate((raw[good, k], bank_raw[use]))
        coeff.append(np.linalg.lstsq(_sensor_features(x, k), y, rcond=None)[0])
    return np.asarray(coeff)


def _calibrate(study):
    from scipy.optimize import root

    c = _sensor_fit(study['sensor_calibration_sweeps'])
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    def forward(x):
        return np.stack([_sensor_features(x, k) @ c[k] for k in range(4)], axis=-1)
    linear = np.stack([forward(e) - c[:, 0] - c[:, 5] * e for e in np.eye(4)], axis=1)
    init = np.linalg.solve(linear, (observed - c[:, 0]).T).T
    calibrated = []
    for y, x0 in zip(observed, init):
        sol = root(lambda x: forward(x) - y, x0, tol=1e-11)
        if np.linalg.norm(sol.fun, np.inf) > 1e-9:
            raise RuntimeError('rosette inversion did not converge')
        calibrated.append(sol.x)
    return np.asarray(calibrated).ravel()


def _axis(kappa, vel, h, leakage):
    n = len(kappa)
    face = (kappa[1:] + kappa[:-1]) * (0.5 / h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(vel[1:-1], 0) / h
    upper[1:] -= np.maximum(-vel[1:-1], 0) / h
    upper[0] = -kappa[0] / h**2
    lower[-1] = -kappa[-1] / h**2
    diag = np.empty(n)
    diag[1:-1] = -lower[:-1] - upper[1:]
    diag[0] = -upper[0] + leakage[0] / h
    diag[-1] = -lower[-1] + leakage[1] / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    # All four walls follow the open Robin law in this campaign.
    nx, ny = int(study['nx']), int(study['ny'])
    ax = _axis(study['kappa_x'], study['vel_x'], study['Lx'] / (nx - 1), (study['robin_x'], study['robin_x']))
    ay = _axis(study['kappa_y'], study['vel_y'], study['Ly'] / (ny - 1), (study['robin_y'], study['robin_y']))
    L = (sparse.kron(sparse.eye(ny), ax, format='csc')
         + sparse.kron(ay, sparse.eye(nx), format='csc')
         + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    ij = np.asarray(study['sensor_ij'])
    rows = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    sampled = []
    for shift in study['mu_shifts']:
        sol = splu(L + shift * eye).solve(rhs)
        sampled.append(sol[rows])
    return np.concatenate(sampled, axis=0), sol.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(study, S, y):
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    values = np.column_stack((S, y))
    values -= z @ np.linalg.solve(gram, z.T @ values)
    return values[:, :-1], values[:, -1]


def _box_simplex(H, f, cap, budget):
    """Primal active-set method with exact minimization on each free face."""
    n = len(f)
    if budget <= 0 or budget >= n * cap:
        w = np.full(n, 0.0 if budget <= 0 else cap)
        grad = H @ w + f
        return w, float(grad.min() if budget <= 0 else grad.max())
    w = np.full(n, budget / n)
    state = np.zeros(n, dtype=int)
    tol = 32 * np.finfo(float).eps * max(1.0, cap)
    for _ in range(20 * n * n):
        free = np.flatnonzero(state == 0)
        fixed = np.flatnonzero(state != 0)
        if len(free):
            A = H[np.ix_(free, free)]
            rhs = -f[free] - H[np.ix_(free, fixed)] @ w[fixed]
            cf = np.linalg.solve(A, np.column_stack((rhs, np.ones(len(free)))))
            nu = (budget - w[fixed].sum() - cf[:, 0].sum()) / cf[:, 1].sum()
            dest = cf[:, 0] + nu * cf[:, 1]
            delta = dest - w[free]
            hits = np.full(len(free), np.inf)
            down = dest < -tol
            up = dest > cap + tol
            hits[down] = -w[free[down]] / delta[down]
            hits[up] = (cap - w[free[up]]) / delta[up]
            j = int(np.argmin(hits))
            if hits[j] < 1.0:
                w[free] += max(0.0, hits[j]) * delta
                state[free[j]] = -1 if down[j] else 1
                w[free[j]] = 0.0 if down[j] else cap
                continue
            w[free] = dest
        grad = H @ w + f
        if not len(free):
            lo, hi = np.flatnonzero(state < 0), np.flatnonzero(state > 0)
            a, b = lo[np.argmin(grad[lo])], hi[np.argmax(grad[hi])]
            if grad[a] >= grad[b] - tol:
                return w, float((grad[a] + grad[b]) / 2)
            state[[a, b]] = 0
            continue
        violation = state * (grad - nu)
        j = int(np.argmax(violation))
        if violation[j] <= tol * max(1.0, np.max(np.abs(grad))):
            return np.clip(w, 0.0, cap), float(nu)
        state[j] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    S, U_last = _forward(study)
    S, y = _project(study, S, _calibrate(study))
    cap, budget, ridge = (float(study[key]) for key in ('u_box', 'total_budget', 'source_ridge'))
    w, nu = _box_simplex(S.T @ S + ridge * np.eye(S.shape[1]), -S.T @ y, cap, budget)
    res = S @ w - y
    g = S.T @ res + ridge * w
    obj = float((res @ res + ridge * (w @ w)) / 2)
    d = g - nu
    dual = obj - g @ w + nu * budget + cap * np.minimum(0., d).sum()
    return {'weights': w,
            'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0., d).tolist(),
                                'lam_upper': np.maximum(0., -d).tolist(),
                                'primal_objective': obj, 'dual_lower_bound': float(dual)}}
