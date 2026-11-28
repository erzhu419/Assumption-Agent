"""Steady transport-reaction inversion for the sealed chamber."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy import sparse
from scipy.optimize import root
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h):
    """Apply the same upwind flux stencil at interior and wrapped nodes."""
    n = len(kappa)
    i = np.arange(n)
    left = (i - 1) % n
    right = (i + 1) % n
    west = -(kappa + kappa[left]) / (2 * h**2) - np.maximum(velocity, 0) / h
    east = -(kappa + kappa[right]) / (2 * h**2) - np.maximum(-velocity, 0) / h
    return sparse.csc_matrix((np.r_[west, -west-east, east],
                             (np.tile(i, 3), np.r_[left, i, right])), shape=(n, n))


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx'] / (nx-1))
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly'] / (ny-1))
    reaction = np.asarray(study['V_reaction_2d']).ravel(order='F')
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(reaction, format='csc'))
    modes = np.asarray(study['source_modes'])
    rhs = modes.reshape(-1, nx*ny, order='F').T
    ij = np.asarray(study['sensor_ij'], dtype=int)
    sensors = ij[:, 0] + nx * ij[:, 1]
    eye = sparse.eye(nx*ny, format='csc')
    readings = []
    for mu in study['mu_shifts']:
        response = splu(operator + mu * eye).solve(rhs)
        readings.append(response[sensors])
    return np.concatenate(readings), response.T.reshape(-1, nx, ny, order='F')


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


def _basis(x):
    return (chebvander(np.asarray(x) / 1.8, 12) - chebvander(0., 12))[..., 1:]


def calibration_map(sweeps, held_out=()):
    ref = np.asarray(sweeps['ref_rosette'])
    raw = np.asarray(sweeps['raw_rosette'])
    single = sweeps['single_bank_sweeps']
    models = []
    for j in range(4):
        sel = np.asarray(single['bank_id']) == j
        self_ref = np.r_[np.asarray(single['ref_signal'])[sel], ref[:, j]]
        other_ref = np.r_[np.zeros(sel.sum()), ref[:, (j+2) % 4]]
        obs = np.r_[np.asarray(single['raw_readout'])[sel], raw[:, j]]
        # Gross readout spikes are well separated from the physical sweep range.
        keep = abs(obs) < 5
        keep[sel.sum() + np.asarray(held_out, dtype=int)] = False
        a = np.column_stack((np.ones(keep.sum()), _basis(self_ref[keep])[:, ::2],
                             _basis(other_ref[keep])))
        models.append(np.linalg.lstsq(a, obs[keep], rcond=None)[0])
    return np.array(models)


def raw_map(x, models):
    b = _basis(x)
    return (models[:, 0] + np.sum(b[..., ::2] * models[:, 1:7], axis=-1)
            + np.sum(np.roll(b, 2, axis=-2) * models[:, 7:], axis=-1))


def calibrate(study):
    models = calibration_map(study['sensor_calibration_sweeps'])
    obs = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    return np.array([root(lambda x: raw_map(x, models) - t, np.zeros(4),
                          tol=1e-11).x for t in obs]).ravel()


def solve_batch(study: dict) -> dict:
    # Sealed endpoints retain the documented normal diffusion and have no wall loss.
    S, U = chamber_response(study)
    w, cert = _inverse(study, S, calibrate(study))
    return {'weights': w, 'terminal_field': np.tensordot(w, U, axes=(0, 0)),
            'kkt_certificate': cert}
