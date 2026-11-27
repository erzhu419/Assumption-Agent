"""Calibrated tracer inversion with a box-simplex optimality certificate."""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(vel, dtype=float) / h
    r = np.broadcast_to(np.asarray(robin, dtype=float), (2,))
    faces = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    diag = np.r_[faces, 0.] + np.r_[0., faces] + abs(v) + diag_add
    kb = k[[0, -1]]
    np.add.at(diag, [0, n-1], r * kb / (h * (r*h/2 + kb)))
    return diags((-faces - np.maximum(v[1:], 0.), diag,
                  -faces - np.maximum(-v[:-1], 0.)), (-1, 0, 1), format='csc')


def _calibrate(s):
    sweeps = s['sensor_calibration_sweeps']
    groups = np.asarray(sweeps['bank_id'])
    y = np.asarray(s['y_obs_raw'], dtype=float).copy()
    bank = np.asarray(s['row_bank_id'])
    for b in np.unique(bank):
        x = np.asarray(sweeps['ref_signal'], dtype=float)[groups == b]
        z = np.asarray(sweeps['raw_readout'], dtype=float)[groups == b]
        i, j = np.triu_indices(len(x), 1)
        dx = x[j] - x[i]
        good = dx != 0
        if not np.any(good):
            raise ValueError(f'Bank {b} has no calibration span')
        # Seed with a median slope so isolated EMI spikes do not set the gain.
        slope = np.median((z[j] - z[i])[good] / dx[good])
        err = z - slope*x
        err -= np.median(err)
        scale = 1.4826 * np.median(abs(err))
        inlier = abs(err) <= max(6*scale, 1e-12 * max(1., np.max(abs(z))))
        if np.ptp(x[inlier]) == 0:
            inlier = np.ones(len(x), dtype=bool)
        gain, offset = np.linalg.lstsq(np.column_stack((x[inlier], np.ones(inlier.sum()))), z[inlier], rcond=None)[0]
        y[bank == b] = (y[bank == b] - offset) / gain
    return y


def _design_and_terminal(s):
    nx, ny = s['nx'], s['ny']
    A = _build_1d(s['Lx'], nx, s['kappa_x'], s['vel_x'], s['robin_x'], np.asarray(s['reaction_x']))
    B = _build_1d(s['Ly'], ny, s['kappa_y'], s['vel_y'], s['robin_y'], np.asarray(s['reaction_y']))
    # The public reference uses the axial reactions plus the 2-D potential.
    op = (kron(A, eye(ny), format='csc') + kron(eye(nx), B, format='csc')
          + diags(np.asarray(s['V_reaction_2d']).ravel(), format='csc'))
    identity = eye(nx*ny, format='csc')
    src = np.asarray(s['source_modes'], dtype=float).reshape(-1, nx*ny).T
    i, j = np.asarray(s['sensor_ij']).T
    rows = i*ny + j
    blocks = []
    for mu in s['mu_shifts']:
        fields = splu(op + mu*identity).solve(src)
        blocks.append(fields[rows])
    return np.concatenate(blocks, axis=0), fields


def _profile(Z, values, ridge):
    if ridge == 0:
        return values - Z @ np.linalg.lstsq(Z, values, rcond=None)[0]
    return values - Z @ np.linalg.solve(Z.T @ Z + ridge*np.eye(Z.shape[1]), Z.T @ values)


def _box_simplex(S, y, ridge, cap, budget):
    m = S.shape[1]
    H = S.T @ S + ridge*np.eye(m)
    c = S.T @ y
    w = np.full(m, budget/m)
    state = np.zeros(m, dtype=int)  # -1 lower, 0 free, +1 upper
    if budget == 0:
        w.fill(0.)
        state.fill(-1)
    elif budget == m*cap:
        w.fill(cap)
        state.fill(1)
    tol = 1e-12 * max(1., cap)
    for _ in range(20*m*m):
        free = np.flatnonzero(state == 0)
        fixed = np.flatnonzero(state != 0)
        if free.size:
            n = free.size
            K = np.zeros((n+1, n+1))
            K[:n, :n] = H[np.ix_(free, free)]
            K[:n, n] = -1.
            K[n, :n] = 1.
            rhs = np.r_[c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
            sol = np.linalg.solve(K, rhs)
            trial, nu = sol[:-1], sol[-1]
            bad = (trial < -tol) | (trial > cap + tol)
            if np.any(bad):
                f = free[bad]
                target = np.where(trial[bad] < 0., 0., cap)
                steps = (target - w[f]) / (trial[bad] - w[f])
                k = np.argmin(steps)
                w[free] += np.clip(steps[k], 0., 1.) * (trial - w[free])
                w[f[k]] = target[k]
                state[f[k]] = -1 if target[k] == 0 else 1
                continue
            w[free] = np.clip(trial, 0., cap)
        grad = H @ w - c
        if not free.size:
            lower = np.flatnonzero(state == -1)
            upper = np.flatnonzero(state == 1)
            if not upper.size or not lower.size:
                nu = grad.min() if not upper.size else grad.max()
                break
            il = lower[np.argmin(grad[lower])]
            iu = upper[np.argmax(grad[upper])]
            if grad[il] >= grad[iu] - 1e-12:
                nu = (grad[il] + grad[iu])/2
                break
            state[[il, iu]] = 0
            continue
        violation = state * (grad - nu)
        k = np.argmax(violation)
        if violation[k] <= 1e-12:
            break
        state[k] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')
    residual = S @ w - y
    grad = S.T @ residual + ridge*w
    p = .5 * (residual @ residual + ridge*(w @ w))
    lb = p - grad @ w + nu*budget + cap*np.minimum(grad-nu, 0.).sum()
    return w, {'weights': w.copy(), 'nu_sum': float(nu),
               'lam_lower': np.where(state == -1, np.maximum(grad-nu, 0.), 0.),
               'lam_upper': np.where(state == 1, np.maximum(nu-grad, 0.), 0.),
               'primal_objective': float(p), 'dual_lower_bound': float(min(lb, p))}


def solve_beta_m96_w01_s00(study: dict) -> dict:
    """Recover source weights and the terminal field from a laminar study."""
    s = study
    D, terminal = _design_and_terminal(s)
    profiled = _profile(np.asarray(s['Z_nuisance']), np.column_stack((D, _calibrate(s))), s['nuisance_ridge'])
    w, cert = _box_simplex(profiled[:, :-1], profiled[:, -1], s['source_ridge'], s['u_box'], s['total_budget'])
    return {'weights': w, 'terminal_field': (terminal @ w).reshape(s['nx'], s['ny']),
            'kkt_certificate': cert}
