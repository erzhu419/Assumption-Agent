"""Forward transport and box-simplex source inversion."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu


def _axis_operator(kappa, velocity, robin, length):
    n = len(kappa)
    h = length / (n - 1)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = np.r_[-face[:-1] - np.maximum(velocity[1:-1], 0) / h,
                  -kappa[-1] / h**2]
    upper = np.r_[-kappa[0] / h**2,
                  -face[1:] - np.maximum(-velocity[1:-1], 0) / h]
    diag = np.r_[kappa[0] / h**2 + robin / h,
                 -lower[:-1] - upper[1:],
                 kappa[-1] / h**2 + robin / h]
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax, ay = [_axis_operator(study['kappa_' + a], study['vel_' + a],
                             study['robin_' + a], study['L' + a]) for a in 'xy']
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(study['V_reaction_2d'].ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    rhs = np.column_stack([mode.ravel(order='F') for mode in study['source_modes']])
    i, j = study['sensor_ij'].T
    sample = i + nx * j
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(operator + shift * identity).solve(rhs)
        blocks.append(fields[sample])
    return np.vstack(blocks), fields.T.reshape(-1, ny, nx).transpose(0, 2, 1)


def _calibrated_observations(study):
    sweep = study['sensor_calibration_sweeps']
    ref = sweep['ref_rosette']
    offset, *mixing = np.linalg.lstsq(np.c_[np.ones(len(ref)), ref],
                                      sweep['raw_rosette'], rcond=None)[0]
    raw = study['y_obs_raw'].reshape(-1, 4)
    return np.linalg.solve(np.asarray(mixing).T, (raw - offset).T).T.ravel()


def _box_simplex(H, c, cap, budget):
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(c.min())
    if budget == n * cap:
        w = np.full(n, cap)
        return w, float((H @ w + c).max())
    ones = np.ones(n)
    scale = max(np.linalg.norm(H, ord=np.inf), np.linalg.norm(c, ord=np.inf))
    if scale == 0:
        return np.full(n, budget / n), 0.0
    result = minimize(lambda w: ((0.5 * w @ H @ w + c @ w) / scale,
                                 (H @ w + c) / scale),
                      np.full(n, budget / n), jac=True, method='SLSQP',
                      bounds=[(0, cap)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - budget,
                                    'jac': lambda w: ones}],
                      options={'ftol': 1e-14, 'maxiter': 500})
    w = result.x
    tol = 1e-9 * cap
    lo, hi = w <= tol, w >= cap - tol
    free = ~(lo | hi)
    if np.any(free):
        # Remove the numerical optimization tolerance on the identified face.
        polished = np.where(hi, cap, 0.0)
        count = int(free.sum())
        a = np.ones(count)
        kkt = np.block([[H[np.ix_(free, free)] / scale, -a[:, None]],
                        [a[None, :], np.zeros((1, 1))]])
        rhs = np.r_[-(H @ polished + c)[free] / scale, budget - polished.sum()]
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        polished[free] = solution[:-1]
        nu = solution[-1] * scale
        reduced = H @ polished + c - nu
        if (np.all((polished >= 0) & (polished <= cap))
                and np.all(reduced[lo] >= -1e-11 * scale)
                and np.all(reduced[hi] <= 1e-11 * scale)):
            return polished, float(nu)
    g = H @ w + c
    if np.any(free):
        nu = np.mean(g[free])
    else:
        nu = 0.5 * (g[hi].max() + g[lo].min())
    return w, float(nu)


def solve_batch(study: dict) -> dict:
    S, terminal_modes = _forward(study)
    y = _calibrated_observations(study)
    Z = study['Z_nuisance']
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    projected = np.column_stack((S, y))
    projected -= Z @ np.linalg.solve(gram, Z.T @ projected)
    design, target = projected[:, :-1], projected[:, -1]
    ridge = study['source_ridge']
    cap, budget = study['u_box'], study['total_budget']
    H = design.T @ design + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, -(design.T @ target), cap, budget)
    residual = design @ w - target
    g = design.T @ residual + ridge * w
    reduced = g - nu
    lower = np.maximum(reduced, 0)
    upper = np.maximum(-reduced, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * upper.sum())
    return {'weights': w,
            'terminal_field': np.tensordot(w, terminal_modes, axes=(0, 0)),
            'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                                'lam_lower': lower.tolist(), 'lam_upper': upper.tolist(),
                                'primal_objective': objective, 'dual_lower_bound': dual}}
