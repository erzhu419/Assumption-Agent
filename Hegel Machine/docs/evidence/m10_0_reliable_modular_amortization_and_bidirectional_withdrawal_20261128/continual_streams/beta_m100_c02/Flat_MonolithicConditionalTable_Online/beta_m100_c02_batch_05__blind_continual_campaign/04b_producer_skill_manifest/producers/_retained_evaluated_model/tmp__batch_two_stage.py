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


def calibration_map(sweeps, held_out=None, degree=10):
    from numpy.polynomial.chebyshev import chebvander
    x = np.asarray(sweeps['ref_rosette'])
    y = np.asarray(sweeps['raw_rosette'])
    bank = sweeps['single_bank_sweeps']
    t, v, ids = (np.asarray(bank[k]) for k in ('ref_signal', 'raw_readout', 'bank_id'))
    c = []
    for i in range(4):
        mask = (ids == i) & (abs(v) < 5.)
        c.append(np.linalg.lstsq(odd_basis(t[mask]), v[mask], rcond=None)[0])
    c = np.array(c)
    gain = c[:, 1]
    own = np.einsum('nik,ik->ni', odd_basis(x), c)
    opposite = np.roll(x, 2, axis=1)
    z = opposite/1.8
    cross = (chebvander(z, degree) - chebvander(0., degree))[..., 1:]
    keep = abs(y) < 5.
    if held_out is not None:
        keep[held_out] = False
    mix = np.linalg.lstsq(cross[keep], ((y-own)/gain)[keep], rcond=None)[0]
    return c, mix


def odd_basis(t):
    z = t/1.5
    return np.stack([np.ones_like(z), z, z**3, z**5, z**7, z**9], axis=-1)


def raw_map(x, c, mix):
    from numpy.polynomial.chebyshev import chebval
    z = np.roll(x, 2, axis=1)/1.8
    cross = chebval(z, np.r_[0., mix]) - chebval(0., np.r_[0., mix])
    return np.einsum('nik,ik->ni', odd_basis(x), c) + c[:, 1]*cross


def calibrate(study):
    c, mix = calibration_map(study['sensor_calibration_sweeps'])
    raw = np.asarray(study['y_obs_raw']).reshape(-1, 4)
    x = (raw-c[:, 0])/(c[:, 1]/1.5)
    for _ in range(15):
        error = raw_map(x, c, mix) - raw
        if np.max(abs(error)) < 1e-13:
            break
        # The Jacobian couples only opposing channels.
        d = (raw_map(x+1e-5, c, mix)-raw_map(x-1e-5, c, mix))/(2e-5)
        a = (raw_map(x+1e-5*np.array([1,1,-1,-1]), c, mix)-raw_map(x-1e-5*np.array([1,1,-1,-1]), c, mix))/(2e-5)
        a *= [1,1,-1,-1]
        diag, off = (d+a)*.5, (d-a)*.5
        det = diag*np.roll(diag, 2, axis=1) - off*np.roll(off, 2, axis=1)
        x -= (np.roll(diag, 2, axis=1)*error - off*np.roll(error, 2, axis=1))/det
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
