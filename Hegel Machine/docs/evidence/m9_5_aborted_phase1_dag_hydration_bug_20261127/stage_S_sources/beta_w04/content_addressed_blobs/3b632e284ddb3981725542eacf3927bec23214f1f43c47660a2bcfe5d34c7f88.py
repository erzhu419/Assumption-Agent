import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _build_1d(length, n, kappa, vel, robin, diag_add):
    h = length / n
    kappa = np.asarray(kappa, dtype=float)
    vel = np.asarray(vel, dtype=float)
    d = 2 * kappa[:-1] * kappa[1:] / ((kappa[:-1] + kappa[1:]) * h**2)
    # Cell-centered upwind derivative, with zero exterior inflow.
    plus, minus = np.maximum(vel, 0) / h, np.minimum(vel, 0) / h
    diag = np.array(diag_add, dtype=float, copy=True) + plus - minus
    diag[:-1] += d
    diag[1:] += d
    boundary = np.array([0, n - 1])
    np.add.at(diag, boundary, robin * kappa[boundary] / (h * (robin * 0.5 * h + kappa[boundary])))
    plus, minus = plus[1:], minus[:-1]
    return sparse.diags((-d - plus, diag, -d + minus), (-1, 0, 1), format='csc')


def _forward(study):
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'], study['robin_x'], study['reaction_x'] + v.mean(1))
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'], study['robin_y'], study['reaction_y'] + v.mean(0) - v.mean())
    identity = sparse.eye(nx * ny, format='csc')
    op = sparse.kron(ax, sparse.eye(ny), format='csc') + sparse.kron(sparse.eye(nx), by, format='csc')
    rhs = np.asfortranarray(modes.reshape(m, -1).T)
    ij = np.asarray(study['sensor_ij'], dtype=int)
    indices = ij[:, 0] * ny + ij[:, 1]
    blocks = []
    for mu in study['mu_shifts']:
        u = splu(op + mu * identity).solve(rhs)
        blocks.append(u[indices])
    return np.concatenate(blocks, axis=0), u.T.reshape(m, nx, ny)


def _robust_affine(x, y, gain=1.0, offset=0.0):
    if x.size == 0:
        return gain, offset
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    good = np.abs(dx) > 1e-12
    if not np.any(good):
        return gain, float(np.median(y - gain * x))
    gain = np.median((y[j[good]] - y[i[good]]) / dx[good])
    offset = np.median(y - gain * x)
    r = y - gain * x - offset
    mask = np.abs(r) <= max(1e-7, 5 * 1.4826 * np.median(np.abs(r)))
    if np.count_nonzero(mask) >= 2 and np.ptp(x[mask]) > 1e-12:
        gain, offset = np.linalg.lstsq(np.column_stack((x[mask], np.ones(mask.sum()))), y[mask], rcond=None)[0]
    return gain, offset


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    x = np.asarray(sweeps['ref_signal'], dtype=float)
    y = np.asarray(sweeps['raw_readout'], dtype=float)
    params = np.asarray([_robust_affine(x[banks == b], y[banks == b]) for b in range(4)])
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (study['y_obs_raw'] - params[rows, 1]) / params[rows, 0]


def _profile(s, y, z, ridge):
    data = np.column_stack((s, y))
    left, singular, _ = np.linalg.svd(z, full_matrices=False)
    if ridge:
        factor = singular**2 / (singular**2 + ridge)
    else:
        factor = singular > np.finfo(float).eps * max(z.shape) * singular.max(initial=0)
    data -= left @ (factor[:, None] * (left.T @ data))
    return data[:, :-1], data[:, -1]


def _optimize(s, y, ridge, cap, budget):
    m = s.shape[1]
    hess = s.T @ s + ridge * np.eye(m)
    linear = s.T @ y
    w = np.full(m, budget / m)
    active = np.zeros(m, dtype=int)
    if budget <= 0:
        w[:] = 0
        nu = float(np.min(-linear))
    elif budget >= m * cap:
        w[:] = cap
        nu = float(np.max(hess @ w - linear))
    else:
        for _ in range(20 * m):
            g = hess @ w - linear
            idx = np.flatnonzero(active == 0)
            if idx.size == 0:
                lo = np.flatnonzero(active == -1)
                hi = np.flatnonzero(active == 1)
                i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
                nu = float((g[i] + g[j]) / 2)
                if g[i] >= g[j] - 1e-13:
                    break
                active[[i, j]] = 0
                continue
            n = idx.size
            system = np.empty((n + 1, n + 1))
            system[:n, :n] = hess[np.ix_(idx, idx)]
            system[:n, n] = -1
            system[n, :n] = 1
            system[n, n] = 0
            p = np.linalg.solve(system, np.r_[-g[idx], budget - w.sum()])
            trial, nu = p[:-1], float(p[-1])
            step = np.full(n, np.inf)
            np.divide(cap - w[idx], trial, out=step, where=trial > 1e-14)
            np.divide(-w[idx], trial, out=step, where=trial < -1e-14)
            j = int(np.argmin(step))
            alpha = np.clip(step[j], 0, 1)
            w[idx] += alpha * trial
            if alpha < 1:
                k = idx[j]
                active[k] = 1 if trial[j] > 0 else -1
                w[k] = cap if active[k] == 1 else 0
                continue
            violation = active * (hess @ w - linear - nu)
            k = int(np.argmax(violation))
            if violation[k] <= 1e-13:
                break
            active[k] = 0
        else:
            raise RuntimeError('box-simplex active set did not converge')
    w = np.clip(w, 0, cap)
    r = s @ w - y
    obj = 0.5 * (r @ r + ridge * (w @ w))
    g = s.T @ r + ridge * w
    d = g - nu
    lower = obj - g @ w + nu * budget + cap * np.minimum(0, d).sum()
    return dict(weights=w, nu_sum=nu, lam_lower=np.maximum(d, 0), lam_upper=np.maximum(-d, 0),
                primal_objective=float(obj), dual_lower_bound=float(min(obj, lower)))


def solve_beta_w04_s00(study: dict) -> dict:
    s, terminal_modes = _forward(study)
    y = _calibrate(study)
    s, y = _profile(s, y, study['Z_nuisance'], study['nuisance_ridge'])
    cert = _optimize(s, y, study['source_ridge'], study['u_box'], study['total_budget'])
    w = cert['weights']
    return dict(weights=w, terminal_field=np.einsum('m,mxy->xy', w, terminal_modes), kkt_certificate=cert)
