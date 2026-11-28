import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares, minimize, root


def _axis(kap, vel, h, robin):
    """Row-local upwind transport with open-wall Robin endpoints."""
    faces = (kap[1:] + kap[:-1]) / (2 * h**2)
    left = -faces - np.maximum(vel[1:], 0) / h
    right = -faces - np.maximum(-vel[:-1], 0) / h
    left[-1] = -kap[-1] / h**2
    right[0] = -kap[0] / h**2
    diag = -np.r_[0., left] - np.r_[right, 0.]
    diag[[0, -1]] += robin / h
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


def _response(x, q):
    a, c = x[..., 0], x[..., 2]
    offset, amp, rate, shoulder, onset, cross, asym, bend, mix = q
    return (offset + amp * np.tanh(rate * a)
            + shoulder * np.maximum(a - onset, 0.)**3
            + cross * c + asym * c**2 / (1. + bend * c**2) + mix * a * c)


def _fit_bank(x, y):
    # Grossly corrupted readouts are outside the calibrated response range.
    keep = np.abs(y) < 5.
    x, y = x[keep], y[keep]
    initial = [0., 1.5, 1.17, .4, 1.05, .25, -.15, .6, 0.]
    bounds = ([-1., .3, .4, 0., .6, 0., -1., 0., -.15],
              [1., 3., 3., 20., 1.49, 2., 0., 5., .15])
    return least_squares(lambda q: _response(x, q) - y, initial, bounds=bounds,
                         loss='soft_l1', f_scale=.0003, max_nfev=350).x


def _calibration(sweeps):
    refs, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    single = sweeps['single_bank_sweeps']
    order = (np.arange(4)[:, None] + np.arange(4)) % 4
    coeff = []
    for j, cols in enumerate(order):
        use = single['bank_id'] == j
        x = np.zeros((np.count_nonzero(use), 4))
        x[:, 0] = single['ref_signal'][use]
        coeff.append(_fit_bank(np.vstack((refs[:, cols], x)),
                               np.r_[raw[:, j], single['raw_readout'][use]]))
    return np.array(coeff)


def _calibrate(study):
    coeff = _calibration(study['sensor_calibration_sweeps'])
    order = (np.arange(4)[:, None] + np.arange(4)) % 4

    def invert(y):
        def residual(x):
            return np.array([_response(x[cols], q) for cols, q in zip(order, coeff)]) - y

        initial = (y - coeff[:, 0]) / (coeff[:, 1] * coeff[:, 2])
        answer = root(residual, initial, tol=1e-10)
        if np.max(np.abs(answer.fun)) > 1e-8:
            raise RuntimeError('Could not invert rosette response')
        return answer.x

    raw = study['y_obs_raw'].reshape(-1, 4)
    return np.array([invert(y) for y in raw]).ravel()


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
