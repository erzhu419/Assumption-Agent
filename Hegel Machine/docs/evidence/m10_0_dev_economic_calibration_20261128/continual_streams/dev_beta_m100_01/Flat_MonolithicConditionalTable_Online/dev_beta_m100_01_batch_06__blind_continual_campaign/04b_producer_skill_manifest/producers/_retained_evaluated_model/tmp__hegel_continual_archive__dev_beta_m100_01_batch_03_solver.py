import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, root


def _axis(kap, vel, h, robin):
    # Midpoint face diffusivities and row-local upwind velocities.
    n = len(kap)
    faces = (kap[1:] + kap[:-1]) / (2 * h*h)
    left = -faces - np.maximum(vel[1:], 0) / h
    right = -faces - np.maximum(-vel[:-1], 0) / h
    left[-1] = -kap[-1] / (h*h)
    right[0] = -kap[0] / (h*h)
    diag = -np.r_[0., left] - np.r_[right, 0.]
    diag[[0, n-1]] += robin / h
    return sparse.diags((left, diag, right), (-1, 0, 1), format='csc')


def _operator(st):
    nx, ny = st['nx'], st['ny']
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx'] / (nx - 1), st['robin_x'])
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly'] / (ny - 1), st['robin_y'])
    return (sparse.kron(sparse.eye(ny), ax, format='csc') +
            sparse.kron(ay, sparse.eye(nx), format='csc') +
            sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))


def _forward(st, operator=None):
    nx, ny = st['nx'], st['ny']
    L = _operator(st) if operator is None else operator
    I = sparse.eye(nx * ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx * ny).T
    ij = st['sensor_ij']
    at = ij[:, 0] + nx * ij[:, 1]
    samples = []
    for mu in st['mu_shifts']:
        U = splu(L + mu * I).solve(rhs)
        samples.append(U[at])
    return np.vstack(samples), U.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _design(x, j):
    a, b, c = x[:, j], x[:, (j + 1) % 4], x[:, (j - 1) % 4]
    return np.column_stack((np.ones(len(x)), a, b, c, a * b, a**3))


def _calibration(sweeps):
    x, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    bank = sweeps['single_bank_sweeps']
    coeff = []
    for j in range(4):
        use = bank['bank_id'] == j
        xb = np.zeros((np.count_nonzero(use), 4))
        xb[:, j] = bank['ref_signal'][use]
        A = np.vstack((_design(x, j), _design(xb, j)))
        y = np.r_[raw[:, j], bank['raw_readout'][use]]
        fit = np.linalg.lstsq(A, y, rcond=None)[0]
        # Downweight contaminated pulses, then refit on the coherent sweeps.
        for _ in range(6):
            e = y - A @ fit
            scale = max(1e-8, 1.4826 * np.median(np.abs(e - np.median(e))))
            q = 1 / np.hypot(1, e / (2.5 * scale))
            fit = np.linalg.lstsq(A * q[:, None], y * q, rcond=None)[0]
        e = y - A @ fit
        scale = 1.4826 * np.median(np.abs(e - np.median(e)))
        keep = np.abs(e) < max(8 * scale, 2e-4)
        coeff.append(np.linalg.lstsq(A[keep], y[keep], rcond=None)[0])
    return np.array(coeff)


def _calibrate(st):
    offset, gain, following, preceding, cross, cubic = _calibration(
        st['sensor_calibration_sweeps']).T

    def forward(x):
        b, c = np.roll(x, -1), np.roll(x, 1)
        return offset + gain * x + following * b + preceding * c + cross * x * b + cubic * x**3

    obs = st['y_obs_raw'].reshape(-1, 4)
    return np.array([root(lambda x: forward(x) - y, (y - offset) / gain,
                          tol=1e-11).x for y in obs]).ravel()


def _project(st, S, y):
    Z = st['Z_nuisance']
    ztz = Z.T @ Z + st['nuisance_ridge'] * np.eye(Z.shape[1])
    proj = np.linalg.solve(ztz, Z.T @ np.column_stack((S, y)))
    projected = np.column_stack((S, y)) - Z @ proj
    return projected[:, :-1], projected[:, -1]


def _qp(A, b, ridge, cap, budget):
    n = A.shape[1]
    H, c = A.T @ A + ridge * np.eye(n), A.T @ b
    def grad(w):
        return H @ w - c
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        return w, float(np.min(grad(w)) if budget == 0 else np.max(grad(w)))
    scale = np.linalg.norm(H, ord=np.inf)
    res = minimize(lambda w: (0.5 * w @ H @ w - c @ w) / scale,
                   np.full(n, budget / n), jac=lambda w: grad(w) / scale,
                   bounds=[(0., cap)] * n, method='SLSQP',
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                'jac': lambda w: np.ones(n)},
                   options={'ftol': 1e-13, 'maxiter': 500})
    w = res.x.copy()
    lower, upper = w <= 1e-8 * cap, w >= (1 - 1e-8) * cap
    # Solve the equality-constrained system on the identified face. Accept the
    # refinement only if it improves the certificate and remains feasible.
    free = ~(lower | upper)
    def multiplier(v):
        g = grad(v)
        mask = (v > 1e-9 * cap) & (v < (1-1e-9) * cap)
        if np.any(mask):
            return float(np.mean(g[mask]))
        return float(np.clip(0., np.max(g[v > cap/2]), np.min(g[v <= cap/2])))
    def gap(v, nu):
        d = grad(v) - nu
        return d.clip(0) @ v + (-d).clip(0) @ (cap-v)
    nu = multiplier(w)
    if np.any(free):
        v = np.where(upper, cap, 0.)
        hf = H[np.ix_(free, free)]
        lin = c[free] - H[free] @ v
        x, p = np.linalg.solve(hf, np.column_stack((lin, np.ones(len(lin))))).T
        mu = (budget - v.sum() - x.sum()) / p.sum()
        v[free] = x + mu * p
        if np.min(v) >= 0. and np.max(v) <= cap and gap(v, mu) <= gap(w, nu):
            w, nu = v, float(mu)
    return w, nu


def solve_batch(study: dict) -> dict:
    S, U_last = _forward(study)
    A, b = _project(study, S, _calibrate(study))
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(A, b, ridge, cap, budget)
    r = A @ w - b
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    g = A.T @ r + ridge * w
    d = g - nu
    return {'weights': w, 'terminal_field': np.tensordot(w, U_last, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': np.maximum(0, d).tolist(),
                                'lam_upper': np.maximum(0, -d).tolist(),
                                'primal_objective': J,
                                'dual_lower_bound': float(J - g @ w + nu * budget +
                                                          cap * np.minimum(0, d).sum())}}
