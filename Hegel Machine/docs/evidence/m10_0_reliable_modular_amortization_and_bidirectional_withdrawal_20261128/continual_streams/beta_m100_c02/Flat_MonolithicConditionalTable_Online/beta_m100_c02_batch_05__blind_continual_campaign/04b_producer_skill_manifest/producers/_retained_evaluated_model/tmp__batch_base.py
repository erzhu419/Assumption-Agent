"""Steady chamber forward model and box-simplex ridge inversion."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, *, periodic):
    n = len(kappa)
    west = np.empty(n-1)
    east = np.empty(n-1)
    center = np.empty(n)
    face = (0.5 / h**2) * (kappa[:-1] + kappa[1:])
    v = velocity[1:-1]
    west[:-1] = -face[:-1] - np.maximum(v, 0) / h
    east[1:] = -face[1:] - np.maximum(-v, 0) / h
    center[1:-1] = -west[:-1] - east[1:]
    if not periodic:
        east[0] = -kappa[0] / h**2
        west[-1] = -kappa[-1] / h**2
        center[[0, -1]] = [-east[0], -west[-1]]
        return sparse.diags((west, center, east), (-1, 0, 1), format='csc')

    # The periodic endpoints are ordinary upwind rows with wrapped neighbors.
    wrap = 0.5 * (kappa[0] + kappa[-1]) / h**2
    east[0] = -face[0] - max(-velocity[0], 0) / h
    west[-1] = -face[-1] - max(velocity[-1], 0) / h
    left = -wrap - max(velocity[0], 0) / h
    right = -wrap - max(-velocity[-1], 0) / h
    center[[0, -1]] = [-east[0] - left, -west[-1] - right]
    a = sparse.diags((west, center, east), (-1, 0, 1), format='lil')
    a[0, -1] += left
    a[-1, 0] += right
    return a.tocsc()


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx']/(nx-1), periodic=True)
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly']/(ny-1), periodic=False)
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(reaction, format='csc'))
    rhs = np.asarray(study['source_modes']).reshape(-1, nx*ny, order='F').T
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx*ny, format='csc')
    readings = []
    for mu in study['mu_shifts']:
        response = splu(operator + mu * eye).solve(rhs)
        readings.append(response[sensors])
    return np.concatenate(readings), response.T.reshape(-1, nx, ny, order='F')


def _fit_sweep(features, values):
    """Refit after removing isolated readout spikes from a sweep."""
    keep = np.ones(len(values), dtype=bool)
    for _ in range(6):
        beta = np.linalg.lstsq(features[keep], values[keep], rcond=None)[0]
        error = np.abs(values - features @ beta)
        clean = error <= max(1e-4, 4 * np.median(error[keep]))
        if np.array_equal(clean, keep) or clean.sum() < features.shape[1]:
            return beta
        keep = clean
    return np.linalg.lstsq(features[keep], values[keep], rcond=None)[0]


def calibration_map(sweeps):
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    single = sweeps['single_bank_sweeps']
    coeff = []
    for i in range(4):
        x = ref[:, i]
        left, right = ref[:, (i-1)%4], ref[:, (i+1)%4]
        a = np.column_stack((np.ones_like(x), x, x**3, left, right, x*right))
        select = single['bank_id'] == i
        t = single['ref_signal'][select]
        zero = np.zeros_like(t)
        b = np.column_stack((np.ones_like(t), t, t**3, zero, zero, zero))
        coeff.append(_fit_sweep(np.vstack((a, b)), np.r_[raw[:, i], single['raw_readout'][select]]))
    return np.array(coeff).T


def calibrate(study):
    offset, gain, cubic, left, right, cross = calibration_map(study['sensor_calibration_sweeps'])
    target = np.asarray(study['y_obs_raw']).reshape(-1, 4) - offset
    x = target / gain
    i = np.arange(4)
    for _ in range(30):
        prev, nxt = np.roll(x, 1, axis=1), np.roll(x, -1, axis=1)
        residual = gain*x + cubic*x**3 + left*prev + right*nxt + cross*x*nxt - target
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac = np.zeros((len(x), 4, 4))
        jac[:, i, i] = gain + 3*cubic*x**2 + cross*nxt
        jac[:, i, (i-1)%4] = left
        jac[:, i, (i+1)%4] = right + cross*x
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    return x.ravel()


def box_simplex(h, c, cap, budget):
    """Active-set solve of 1/2 w'Hw - c'w with a fixed sum."""
    n = len(c)
    if budget == 0:
        return np.zeros(n), float(np.min(-c))
    if budget == cap*n:
        w = np.full(n, cap)
        return w, float(np.max(h @ w - c))
    w = np.full(n, budget/n)
    side = np.zeros(n, dtype=int)
    for _ in range(max(100, 20*n)):
        free = np.flatnonzero(side == 0)
        g = h @ w - c
        if not len(free):
            lo = np.where(side < 0, g, np.inf).argmin()
            hi = np.where(side > 0, g, -np.inf).argmax()
            if g[hi] <= g[lo] + 2e-13:
                return w, float((g[hi]+g[lo])/2)
            side[[lo, hi]] = 0
            continue
        hf = h[np.ix_(free, free)]
        z, q = np.linalg.solve(hf, np.column_stack((g[free], np.ones(len(free))))).T
        nu = z.sum() / q.sum()
        d = nu*q-z
        target = w[free] + d
        hits = ((target < -2e-14) | (target > cap+2e-14))
        if hits.any():
            hits = np.flatnonzero(hits)
            dest = np.where(d[hits] < 0, 0., cap)
            alpha = (dest-w[free[hits]])/d[hits]
            j = alpha.argmin()
            w[free] += np.clip(alpha[j], 0., 1.)*d
            k = free[hits[j]]
            w[k] = dest[j]
            side[k] = -1 if dest[j] == 0 else 1
            continue
        w[free] = np.clip(target, 0., cap)
        g = h @ w - c
        nu = g[free].mean()
        violation = side*(g-nu)
        k = violation.argmax()
        if violation[k] <= 2e-13:
            return w, float(nu)
        side[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def _inverse(study, S, y_cal):
    Z = np.asarray(study['Z_nuisance'])
    Pz = np.linalg.solve(Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    S = S - Z @ (Pz @ S)
    r_y = y_cal - Z @ (Pz @ y_cal)
    ridge, cap, budget = study['source_ridge'], study['u_box'], study['total_budget']
    h = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = box_simplex(h, S.T @ r_y, cap, budget)
    r = S @ w - r_y
    g = S.T @ r + ridge * w
    J = float(0.5 * (r @ r + ridge * (w @ w)))
    dual = J - g @ w + nu * budget + cap * np.minimum(0, g - nu).sum()
    return w, {'weights': w.tolist(), 'nu_sum': float(nu),
               'lam_lower': np.maximum(0, g - nu).tolist(),
               'lam_upper': np.maximum(0, nu - g).tolist(),
               'primal_objective': J, 'dual_lower_bound': float(dual)}


def solve_batch(study: dict) -> dict:
    S, U = chamber_response(study)
    w, cert = _inverse(study, S, calibrate(study))
    return {'weights': w, 'terminal_field': np.tensordot(w, U, axes=(0, 0)),
            'kkt_certificate': cert}
