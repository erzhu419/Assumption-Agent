"""Calibrated transport inversion and box-simplex optimality certificate."""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize, root
from scipy.sparse.linalg import splu


def _fit_rosette(sweeps, cross_degree=9):
    x, y = (np.asarray(sweeps[k]) for k in ('ref_rosette', 'raw_rosette'))
    clean = np.max(np.abs(y), axis=1) < 8
    x, y = x[clean], y[clean]
    bank = sweeps['single_bank_sweeps']
    ids, t, z = (np.asarray(bank[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    clean = np.abs(z) < 8
    ids, t, z = ids[clean], t[clean], z[clean]
    opposite = [2, 3, 0, 1]
    polyval = np.polynomial.polynomial.polyval

    def self_response(v, p):
        return v * polyval(v * v, np.r_[1., p[8:14]])

    def predict(v, p):
        return p[:4] + p[4:8] * (self_response(v, p) + polyval(v[..., opposite], np.r_[0., p[14:]]))

    def residual(p):
        return np.r_[(predict(x, p) - y).ravel(), p[:4][ids] + p[4:8][ids] * self_response(t, p) - z]

    start = np.r_[np.zeros(4), np.full(4, 1.5), -.4, .12, -.018, np.zeros(3 + cross_degree)]
    p = least_squares(residual, start, gtol=1e-13).x
    return lambda v: predict(np.asarray(v), p)


def _calibrate(study):
    predict = _fit_rosette(study['sensor_calibration_sweeps'])
    rows = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.concatenate([root(lambda x: predict(x) - y, np.zeros(4), tol=1e-11).x for y in rows])


def _axis(k, v, h, robin, law):
    n = len(k)
    face = (k[:-1] + k[1:]) / (2 * h**2)
    lower = -face.copy()
    upper = -face.copy()
    lower[:-1] -= np.maximum(v[1:-1], 0) / h
    upper[1:] -= np.maximum(-v[1:-1], 0) / h
    diag = np.empty(n)
    diag[1:-1] = -lower[:-1] - upper[1:]
    if law == 'periodic':
        seam = (k[0] + k[-1]) / (2 * h**2)
        wrap_lo = -seam - max(v[0], 0) / h
        wrap_hi = -seam - max(-v[-1], 0) / h
        upper[0] -= max(-v[0], 0) / h
        lower[-1] -= max(v[-1], 0) / h
        diag[0], diag[-1] = -upper[0] - wrap_lo, -lower[-1] - wrap_hi
        return sparse.diags(([wrap_hi], lower, diag, upper, [wrap_lo]), (1 - n, -1, 0, 1, n - 1), format='csc')
    upper[0], lower[-1] = -k[0] / h**2, -k[-1] / h**2
    lo, hi = law
    diag[0], diag[-1] = -upper[0] + lo * robin / h, -lower[-1] + hi * robin / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study, laws):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_axis(study['kappa_' + a], study['vel_' + a], study['L' + a] / (n - 1), study['robin_' + a], law)
              for a, n, law in zip('xy', (nx, ny), laws)]
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc') + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    rhs = np.column_stack([m.ravel(order='F') for m in study['source_modes']])
    ij = study['sensor_ij']
    rows = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx * ny, format='csc')
    sampled = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * eye).solve(rhs)
        sampled.append(fields[rows])
    return np.concatenate(sampled), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _project(Z, ridge, A):
    return A - Z @ np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ A)


def _qp(S, y, ridge, cap, budget):
    n = S.shape[1]
    H, b = S.T @ S + ridge * np.eye(n), S.T @ y
    g = lambda w: H @ w - b
    w = minimize(lambda w: .5 * w @ H @ w - b @ w, np.full(n, budget / n), jac=g,
                 bounds=[(0., cap)] * n,
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget, 'jac': lambda w: np.ones(n)}],
                 method='SLSQP', options={'ftol': 1e-14, 'maxiter': 300}).x
    low, high = w < 1e-8, w > cap - 1e-8
    for _ in range(5*n + 1):
        free = ~(low | high)
        w[low], w[high] = 0., cap
        if np.any(free):
            Hf = H[np.ix_(free, free)]
            ones = np.ones(free.sum())
            s, d = np.linalg.solve(Hf, np.column_stack((b[free] - H[free][:, high].sum(axis=1) * cap, ones))).T
            nu = (budget - w.sum() + w[free].sum() - s.sum()) / d.sum()
            w[free] = s + nu * d
            j = np.argmax(np.where(free, np.maximum(-w, w - cap), -np.inf))
            if w[j] < -1e-12 or w[j] > cap + 1e-12:
                low[j], high[j] = w[j] < 0, w[j] > cap
                continue
        else:
            gg = g(w)
            bottom = np.max(gg[high], initial=-np.inf)
            top = np.min(gg[low], initial=np.inf)
            if bottom > top:
                low[np.argmin(np.where(low, gg, np.inf))] = False
                high[np.argmax(np.where(high, gg, -np.inf))] = False
                continue
            nu = np.clip(0., bottom, top)
        gg = g(w) - nu
        bad = np.where(low, -gg, np.where(high, gg, 0.))
        if np.max(bad) <= 1e-11:
            return w, float(nu)
        j = np.argmax(bad)
        low[j] = high[j] = False
    raise RuntimeError('box-simplex active set did not converge')


def solve_batch(study: dict) -> dict:
    """Reconstruct the sources and certify the constrained optimum."""
    S, fields = _forward(study, ('periodic', 'periodic'))
    y = _calibrate(study)
    A = _project(study['Z_nuisance'], study['nuisance_ridge'], np.column_stack((S, y)))
    S, y = A[:, :-1], A[:, -1]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(S, y, ridge, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    J = .5 * (r @ r + ridge * (w @ w))
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': float(nu),
                                'lam_lower': np.maximum(0., g - nu).tolist(),
                                'lam_upper': np.maximum(0., nu - g).tolist(),
                                'primal_objective': float(J),
                                'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum())}}
