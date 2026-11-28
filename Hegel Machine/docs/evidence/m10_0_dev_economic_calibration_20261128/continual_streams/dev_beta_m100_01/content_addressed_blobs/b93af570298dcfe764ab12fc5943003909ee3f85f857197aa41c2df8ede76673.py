import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares, minimize

_OPP = np.array([2, 3, 0, 1])


def _readout(x, p):
    offset, gain = p[:4], p[4:8]
    amplitude, frequency, mixing, curvature, cross_frequency = p[8:]
    b = x[..., _OPP]
    return offset + gain * (x + amplitude * np.sin(frequency * x) + mixing * b
                            + curvature * (1 - np.cos(cross_frequency * b)))


def _sweeps(st):
    c = st['sensor_calibration_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']
    s = c['single_bank_sweeps']
    return x, y, s['bank_id'], s['ref_signal'], s['raw_readout']


def _fit_calibration(x, y, bank, signal, readout):
    axis = np.eye(4)[bank] * signal[:, None]
    p0 = np.r_[np.zeros(4), np.ones(4), .25, 2.5, .2, -.06, 2.]
    def residual(p):
        return np.r_[(_readout(x, p) - y).ravel(),
                     _readout(axis, p)[np.arange(len(bank)), bank] - readout]
    p = least_squares(residual, p0, loss='soft_l1', f_scale=.005).x
    # Refit at the sweep noise scale after identifying gross corrupted readings.
    keep = np.abs(residual(p)) < .005
    return least_squares(lambda z: residual(z)[keep], p, gtol=1e-12).x


def _calibrate(st):
    p = _fit_calibration(*_sweeps(st))
    raw = st['y_obs_raw'].reshape(-1, 4)
    v = (raw - p[:4]) / p[4:8]
    x = v.copy()
    for _ in range(24):
        b = x[:, _OPP]
        a, f, m, c, h = p[8:]
        r = (x + a * np.sin(f*x) + m*b + c*(1-np.cos(h*b))) - v
        if np.max(abs(r)) < 2e-14:
            break
        d = 1 + a*f*np.cos(f*x)
        e = m + c*h*np.sin(h*b)
        x -= (d[:, _OPP]*r - e*r[:, _OPP]) / (d*d[:, _OPP] - e*e[:, _OPP])
    return x.ravel()


def _axis(kap, vel, h):
    """Periodic midpoint diffusivity and row-local upwind transport."""
    n = len(kap)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    km = (kap + kap[left]) / (2*h*h) + np.maximum(vel, 0)/h
    kp = (kap + kap[right]) / (2*h*h) + np.maximum(-vel, 0)/h
    return sparse.csc_matrix((np.r_[km + kp, -km, -kp],
                              (np.tile(i, 3), np.r_[i, left, right])), shape=(n, n))


def _operator(st):
    nx, ny = st['nx'], st['ny']
    ax = _axis(st['kappa_x'], st['vel_x'], st['Lx']/(nx-1))
    ay = _axis(st['kappa_y'], st['vel_y'], st['Ly']/(ny-1))
    return (sparse.kron(sparse.eye(ny), ax, format='csc')
            + sparse.kron(ay, sparse.eye(nx), format='csc')
            + sparse.diags(st['V_reaction_2d'].ravel(order='F'), format='csc'))


def _forward(st):
    nx, ny = st['nx'], st['ny']
    L = _operator(st)
    I = sparse.eye(nx*ny, format='csc')
    rhs = st['source_modes'].transpose(0, 2, 1).reshape(-1, nx*ny).T
    ij = st['sensor_ij']
    at = ij[:, 0] + nx*ij[:, 1]
    samples = []
    for mu in st['mu_shifts']:
        U = splu(L + mu*I).solve(rhs)
        samples.append(U[at])
    return np.vstack(samples), U.T.reshape(-1, ny, nx).transpose(0, 2, 1)

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
        w = np.full(n, 0. if budget == 0 else cap)
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
