"""Periodic chamber reconstruction with calibrated observations and a KKT certificate."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    design = np.column_stack((np.ones(len(ref)), ref))
    coeff = np.linalg.lstsq(design, raw, rcond=None)[0]
    observed = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.linalg.solve(coeff[1:].T, (observed - coeff[0]).T).T.ravel()


def _periodic_axis(k, v, h):
    """Row-local upwind fluxes on a ring, including the endpoint seam."""
    n = len(k)
    i = np.arange(n)
    left, right = (i - 1) % n, (i + 1) % n
    lower = -(k + k[left]) / (2 * h**2) - np.maximum(v, 0) / h
    upper = -(k + k[right]) / (2 * h**2) - np.maximum(-v, 0) / h
    return sparse.csc_matrix((np.r_[lower, -lower - upper, upper],
                              (np.tile(i, 3), np.r_[left, i, right])), shape=(n, n))


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax, ay = [_periodic_axis(study['kappa_' + a], study['vel_' + a], study['L' + a] / (n - 1))
              for a, n in zip('xy', (nx, ny))]
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
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        return w, float(np.min(g(w)) if budget == 0 else np.max(g(w)))
    # Identify active bounds numerically, then solve the free KKT system directly.
    w = minimize(lambda w: .5 * w @ H @ w - b @ w, np.full(n, budget / n), jac=g,
                 bounds=[(0., cap)] * n,
                 constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget, 'jac': lambda w: np.ones(n)}],
                 method='SLSQP', options={'ftol': 1e-14, 'maxiter': 300}).x
    active_tol = min(1e-8, cap / 4)
    low, high = w < active_tol, w > cap - active_tol
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
            if w.sum() > budget + 1e-12:
                high[np.argmax(np.where(high, gg, -np.inf))] = False
                continue
            if w.sum() < budget - 1e-12:
                low[np.argmin(np.where(low, gg, np.inf))] = False
                continue
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
    raw_S, fields = _forward(study)
    y = _calibrate(study)
    A = _project(study['Z_nuisance'], study['nuisance_ridge'], np.column_stack((raw_S, y)))
    S, y = A[:, :-1], A[:, -1]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _qp(S, y, ridge, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    J = .5 * (r @ r + ridge * (w @ w))
    return {'weights': w, 'terminal_field': np.tensordot(w, fields, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': np.maximum(0., g - nu).tolist(),
                                'lam_upper': np.maximum(0., nu - g).tolist(), 'primal_objective': float(J),
                                'dual_lower_bound': float(J - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum())}}
