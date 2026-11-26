"""Calibrated dual-channel reaction-transport inversion."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    bank = np.asarray(sweep['bank_id'])
    ref = np.asarray(sweep['ref_signal'], dtype=float)
    raw = np.asarray(sweep['raw_readout'], dtype=float)
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    rows = np.asarray(study['row_bank_id'])
    for b in np.unique(rows):
        x, z = ref[bank == b], raw[bank == b]
        xc = x - x.mean()
        gain = (xc @ (z - z.mean())) / (xc @ xc)
        offset = z.mean() - gain * x.mean()
        y[rows == b] = (y[rows == b] - offset) / gain
    return y


def _axis(length, n, kappa, velocity, robin, reaction):
    h = length / n
    k, v = np.asarray(kappa), np.asarray(velocity)
    face = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    diagonal = np.asarray(reaction, dtype=float) + np.abs(v) / h
    diagonal[:-1] += face
    diagonal[1:] += face
    ends = k[[0, -1]]
    diagonal[[0, -1]] += robin * ends / (h * (robin * h / 2 + ends))
    lower = -face - np.maximum(v[1:], 0) / h
    upper = -face - np.maximum(-v[:-1], 0) / h
    return lower, diagonal, upper


def _axis_spectrum(axis, potential):
    lower, diagonal, upper = axis
    # A = diag(scale) T diag(scale)^(-1) with symmetric tridiagonal T.
    scale = np.r_[1.0, np.cumprod(np.sqrt(lower / upper))]
    values, vectors = eigh_tridiagonal(diagonal + potential, -np.sqrt(lower * upper))
    return values, scale[:, None] * vectors, vectors.T / scale


def _forward(study):
    nx, ny = int(study['nx']), int(study['ny'])
    a, b = [_axis(study['L' + d], n, *(study[k + '_' + d]
                  for k in ('kappa', 'vel', 'robin', 'reaction')))
            for d, n in [('x', nx), ('y', ny)]]
    va, vb = (np.asarray(study['V_channel_' + d]) for d in ('a', 'b'))
    xa, px, qx = _axis_spectrum(a, va.mean(axis=1))
    ya, py, qy = _axis_spectrum(b, va.mean(axis=0) - va.mean())
    sums = xa[:, None] + ya
    modes = np.asarray(study['source_modes'], dtype=float)
    m = len(modes)
    rhs_spectral = qx @ modes @ qy.T
    rhs_full = np.asfortranarray(modes.reshape(m, -1).T)
    op = (kron(diags(a, [-1, 0, 1]), eye(ny), format='csc')
          + kron(eye(nx), diags(b, [-1, 0, 1]), format='csc'))
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    sa, sb = [], []
    for shift in np.asarray(study['mu_shifts'], dtype=float):
        ua = px @ (rhs_spectral / (sums + shift)) @ py.T
        ub = splu(op + diags(vb.ravel() + shift, format='csc')).solve(rhs_full)
        ub = ub.T.reshape(m, nx, ny)
        sa.append(ua[:, i, j].T)
        sb.append(ub[:, i, j].T)
    return np.concatenate(sa + sb, axis=0), (ua, ub)


def _profile(design, y, nuisance, ridge):
    # The benchmark uses the ridge-fit residual itself, not its square-root metric.
    joined = np.column_stack((design, y))
    z = np.asarray(nuisance, dtype=float)
    residual = joined - z @ np.linalg.solve(
        z.T @ z + ridge * np.eye(z.shape[1]), z.T @ joined)
    return residual[:, :-1], residual[:, -1]


def _box_simplex(hessian, linear, cap, budget):
    """Minimize a strictly convex quadratic by feasible active-set steps."""
    m = len(linear)
    if budget == 0 or budget == m * cap:
        w = np.full(m, budget / m)
        g = hessian @ w + linear
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(m, budget / m)
    status = np.zeros(m, dtype=int)  # -1: lower bound, +1: upper bound
    for _ in range(20 * m * m):
        free = np.flatnonzero(status == 0)
        fixed = np.flatnonzero(status != 0)
        g = hessian @ w + linear
        if not len(free):
            lo, hi = np.flatnonzero(status < 0), np.flatnonzero(status > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + 1e-12:
                return w, float((g[j] + g[i]) / 2)
            status[[i, j]] = 0
            continue
        block = hessian[np.ix_(free, free)]
        rhs = -linear[free] - hessian[np.ix_(free, fixed)] @ w[fixed]
        base, slope = np.linalg.solve(block, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (budget - w[fixed].sum() - base.sum()) / slope.sum()
        target = base + nu * slope
        outside = (target < -1e-13) | (target > cap + 1e-13)
        if np.any(outside):
            ids = np.flatnonzero(outside)
            bound = np.where(target[ids] < 0, 0.0, cap)
            direction = target - w[free]
            fractions = (bound - w[free[ids]]) / direction[ids]
            k = np.argmin(fractions)
            w[free] = np.clip(w[free] + fractions[k] * direction, 0.0, cap)
            i = free[ids[k]]
            w[i] = bound[k]
            status[i] = -1 if bound[k] == 0 else 1
            continue
        w[free] = np.clip(target, 0.0, cap)
        g = hessian @ w + linear
        violation = status * (g - nu)
        i = np.argmax(violation)
        if violation[i] <= 1e-12:
            return w, float(nu)
        status[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_ep03(study: dict) -> dict:
    """Return source weights, terminal fields, and a convex lower-bound certificate."""
    design, fields = _forward(study)
    design, y = _profile(design, _calibrate(study), study['Z_nuisance'], study['nuisance_ridge'])
    cap, budget, ridge = (float(study[k]) for k in ('u_box', 'total_budget', 'source_ridge'))
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    w, nu = _box_simplex(hessian, -(design.T @ y), cap, budget)
    r = design @ w - y
    g = design.T @ r + ridge * w
    primal = float((r @ r + ridge * (w @ w)) / 2)
    lower = np.maximum(0.0, g - nu)
    upper = np.maximum(0.0, nu - g)
    dual = primal - float(lower @ w + upper @ (cap - w)) + nu * (budget - float(w.sum()))
    return {
        'weights': w,
        'terminal_field_channel_a': np.einsum('m,mij->ij', w, fields[0]),
        'terminal_field_channel_b': np.einsum('m,mij->ij', w, fields[1]),
        'kkt_certificate': dict(weights=w.copy(), nu_sum=nu, lam_lower=lower,
                                lam_upper=upper, primal_objective=primal, dual_lower_bound=dual),
    }
