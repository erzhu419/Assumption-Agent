"""Steady chamber forward model and box-simplex ridge inversion."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def axis_operator(kappa, velocity, h, robin):
    n = len(kappa)
    west = np.empty(n-1)
    east = np.empty(n-1)
    center = np.empty(n)
    face = (0.5 / h**2) * (kappa[:-1] + kappa[1:])
    v = velocity[1:-1]
    west[:-1] = -face[:-1] - np.maximum(v, 0) / h
    east[1:] = -face[1:] - np.maximum(-v, 0) / h
    center[1:-1] = -west[:-1] - east[1:]
    east[0] = -kappa[0] / h**2
    west[-1] = -kappa[-1] / h**2
    center[[0, -1]] = [-east[0] + robin/h, -west[-1] + robin/h]
    return sparse.diags((west, center, east), (-1, 0, 1), format='csc')


def chamber_response(study):
    nx, ny = int(study['nx']), int(study['ny'])
    ax = axis_operator(study['kappa_x'], study['vel_x'], study['Lx']/(nx-1), study['robin_x'])
    ay = axis_operator(study['kappa_y'], study['vel_y'], study['Ly']/(ny-1), study['robin_y'])
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


def calibration_map(sweeps, held_out=None):
    """Jointly fit the isolated and mixed sweeps, sharing the transfer curves."""
    from numpy.polynomial.chebyshev import chebvander
    from scipy.optimize import least_squares

    X = np.asarray(sweeps['ref_rosette'])
    Y = np.asarray(sweeps['raw_rosette'])
    single = sweeps['single_bank_sweeps']
    t, v, ids = (np.asarray(single[k]) for k in ('ref_signal', 'raw_readout', 'bank_id'))
    x = np.r_[t, X.ravel()]
    opposite = np.r_[np.zeros_like(t), np.roll(X, 2, axis=1).ravel()]
    bank = np.r_[ids, np.tile(np.arange(4), len(X))].astype(int)
    obs = np.r_[v, Y.ravel()]
    # The gross sweep spikes are separate from the physical readout range.
    keep = abs(obs) < 5.
    if held_out is not None:
        keep[len(t) + (4*np.asarray(held_out)[:, None] + np.arange(4)).ravel()] = False
    A = chebvander(x/1.8, 9)[:, 1::2]
    B = (chebvander(opposite/1.8, 8) - chebvander(0., 8))[:, 1:]

    def error(p):
        gain = np.r_[1., p[4:7]]
        return (p[bank] + gain[bank]*(A@p[7:12] + B@p[12:]) - obs)[keep]

    p0 = np.r_[np.zeros(4), 1.6, 1., 1.6, 1., np.zeros(12)]
    p = least_squares(error, p0, loss='soft_l1', f_scale=3e-5,
                      gtol=1e-11, max_nfev=200).x
    own = np.zeros(10)
    own[1::2] = p[7:12]
    cross = np.r_[0., p[12:]]
    return p[:4], np.r_[1., p[4:7]], own, cross


def raw_map(x, offset, gain, own, cross):
    from numpy.polynomial.chebyshev import chebval
    z = np.roll(x, 2, axis=1)/1.8
    return offset + gain*(chebval(x/1.8, own) + chebval(z, cross) - chebval(0., cross))


def calibrate(study):
    from numpy.polynomial.chebyshev import chebder, chebval

    offset, gain, own, cross = calibration_map(study['sensor_calibration_sweeps'])
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    target = (raw-offset)/gain
    own_d, cross_d = chebder(own)/1.8, chebder(cross)/1.8
    c0 = chebval(0., cross)
    x = target / chebval(0., own_d)
    for _ in range(15):
        z = np.roll(x, 2, axis=1)/1.8
        error = chebval(x/1.8, own) + chebval(z, cross) - c0 - target
        if np.max(abs(error)) < 1e-13:
            break
        diag, off = chebval(x/1.8, own_d), chebval(z, cross_d)
        paired_diag = np.roll(diag, 2, axis=1)
        det = diag*paired_diag - off*np.roll(off, 2, axis=1)
        x -= (paired_diag*error - off*np.roll(error, 2, axis=1))/det
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
