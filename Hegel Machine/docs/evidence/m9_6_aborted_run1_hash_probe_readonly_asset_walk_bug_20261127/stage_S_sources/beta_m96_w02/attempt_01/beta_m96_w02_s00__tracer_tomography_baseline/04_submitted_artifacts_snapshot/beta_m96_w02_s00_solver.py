"""Calibrated multi-shift finite-volume tracer inversion."""
import numpy as np
from scipy.sparse import diags, eye, kronsum
from scipy.sparse.linalg import splu


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(vel, dtype=float) / h
    face = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    main = np.asarray(diag_add, dtype=float) + np.abs(v)
    main[:-1] += face
    main[1:] += face
    main[0] += robin * k[0] / (h * (robin * h / 2 + k[0]))
    main[-1] += robin * k[-1] / (h * (robin * h / 2 + k[-1]))
    return diags((-face - np.maximum(v[1:], 0), main,
                  -face + np.minimum(v[:-1], 0)), (-1, 0, 1), format='csc')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    if isinstance(sweeps, dict):
        sweeps = [sweeps]
    bank, ref, raw = (np.concatenate([np.broadcast_to(r[key], np.shape(r['ref_signal'])).ravel()
                                      for r in sweeps])
                      for key in ('bank_id', 'ref_signal', 'raw_readout'))
    y = np.asarray(study['y_obs_raw'], dtype=float).copy()
    row_bank = np.asarray(study['row_bank_id'])
    for b in np.unique(row_bank):
        x, z = ref[bank == b], raw[bank == b]
        i, j = np.triu_indices(x.size, 1)
        distinct = x[i] != x[j]
        if not np.any(distinct):
            raise ValueError(f'Bank {b} has no sweep with two distinct reference levels')
        i, j = i[distinct], j[distinct]
        slope = np.median((z[j] - z[i]) / (x[j] - x[i]))
        residual = z - slope * x
        residual -= np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual))
        cutoff = max(5 * scale, 32 * np.finfo(float).eps * max(1.0, np.max(np.abs(z))))
        good = np.abs(residual) <= cutoff
        if np.unique(x[good]).size < 2:
            good = np.ones(x.size, dtype=bool)
        gain, offset = np.linalg.lstsq(np.column_stack((x[good], np.ones(good.sum()))),
                                       z[good], rcond=None)[0]
        y[row_bank == b] = (y[row_bank == b] - offset) / gain
    return y


def _sensor_responses(study):
    V = np.asarray(study['V_reaction_2d'], dtype=float)
    modes = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = modes.shape
    # The nominal axis reactions are in addition to the distributed potential.
    ax = _build_1d(study['Lx'], nx, study['kappa_x'], study['vel_x'],
                   study['robin_x'], study['reaction_x'])
    by = _build_1d(study['Ly'], ny, study['kappa_y'], study['vel_y'],
                   study['robin_y'], study['reaction_y'])
    base = kronsum(by, ax, format='csc') + diags(V.ravel(), format='csc')
    identity = eye(nx * ny, format='csc')
    rhs = modes.reshape(m, -1).T
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(base + mu * identity).solve(rhs).reshape(nx, ny, m)
        blocks.append(fields[i, j])
    return np.concatenate(blocks, axis=0), fields


def _profile(Z, ridge, values):
    q, singular, _ = np.linalg.svd(Z, full_matrices=False)
    d2 = singular**2
    if ridge == 0:
        keep = singular > np.finfo(float).eps * max(Z.shape) * singular.max(initial=0.0)
        q = q[:, keep]
        return values - q @ (q.T @ values)
    return values - q @ ((d2 / (d2 + ridge))[:, None] * (q.T @ values))


def _box_simplex(H, f, upper, budget):
    """Primal active set: minimize 1/2 w' H w - f' w on the capped simplex."""
    n = len(f)
    if budget == 0:
        return np.zeros(n), float(np.min(-f))
    if budget == n * upper:
        w = np.full(n, upper)
        return w, float(np.max(H @ w - f))
    w = np.full(n, budget / n)
    active = np.zeros(n, dtype=int)
    for _ in range(20 * n * n):
        free = np.flatnonzero(active == 0)
        g = H @ w - f
        if free.size == 0:
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] - g[i] <= 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue
        block = H[np.ix_(free, free)]
        rhs = -g[free]
        ones = np.ones(free.size)
        kkt = np.block([[block, -ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
        solution = np.linalg.solve(kkt, np.r_[rhs, budget - w.sum()])
        step, nu = solution[:-1], float(solution[-1])
        target = w[free] + step
        hits = np.flatnonzero((target < -1e-13) | (target > upper + 1e-13))
        if hits.size:
            bounds = np.where(step[hits] < 0, 0.0, upper)
            fraction = (bounds - w[free[hits]]) / step[hits]
            p = int(np.argmin(fraction))
            w[free] += max(0.0, fraction[p]) * step
            index = free[hits[p]]
            w[index] = bounds[p]
            active[index] = -1 if bounds[p] == 0 else 1
            continue
        w[free] = np.clip(target, 0.0, upper)
        violations = active * (H @ w - f - nu)
        p = int(np.argmax(violations))
        if violations[p] <= 1e-12:
            return w, nu
        active[p] = 0
    raise RuntimeError('Box-simplex active-set solve did not converge')


def solve_beta_m96_w02_s00(study: dict) -> dict:
    y = _calibrate(study)
    S, fields = _sensor_responses(study)
    profiled = _profile(np.asarray(study['Z_nuisance']), float(study['nuisance_ridge']),
                        np.column_stack((S, y)))
    S, r = profiled[:, :-1], profiled[:, -1]
    ridge = float(study['source_ridge'])
    u, b = float(study['u_box']), float(study['total_budget'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _box_simplex(H, S.T @ r, u, b)
    res = S @ w - r
    obj = float((res @ res + ridge * (w @ w)) / 2)
    g = S.T @ res + ridge * w
    ll = np.where(w <= 1e-12, np.maximum(g - nu, 0), 0)
    lu = np.where(w >= u - 1e-12, np.maximum(nu - g, 0), 0)
    dual = min(obj, float(obj - g @ w + nu * b + u * np.minimum(g - nu, 0).sum()))
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=ll, lam_upper=lu,
                primal_objective=obj, dual_lower_bound=dual)
    return dict(weights=w, terminal_field=fields @ w, kkt_certificate=cert)
