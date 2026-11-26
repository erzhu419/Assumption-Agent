"""Calibrated tracer transport and box-simplex reconstruction."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    rows = np.asarray(study['row_bank_id'])
    obs = np.asarray(study['y_obs_raw'], dtype=float)
    y = np.empty_like(obs)
    for bank in np.unique(rows):
        keep = banks == bank
        x = ref[keep]
        z = raw[keep]
        dx = x - x.mean()
        gain = (dx @ (z - z.mean())) / (dx @ dx)
        take = rows == bank
        y[take] = (obs[take] - z.mean()) / gain + x.mean()
    return y


def _axis(study, axis):
    n = study['n' + axis]
    h = study['L' + axis] / n
    k = np.asarray(study['kappa_' + axis], dtype=float)
    v = np.asarray(study['vel_' + axis], dtype=float) / h
    rg = study['robin_' + axis]
    d = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    main = np.asarray(study['reaction_' + axis], dtype=float) + abs(v)
    main[:-1] += d
    main[1:] += d
    for i in (0, n - 1):
        main[i] += rg * k[i] / (h * (rg * h / 2 + k[i]))
    return sparse.diags((-d - np.maximum(v[1:], 0), main,
                         -d + np.minimum(v[:-1], 0)), (-1, 0, 1), format='csc')


def _transport(study):
    nx, ny = study['nx'], study['ny']
    ix = sparse.eye(nx, format='csc')
    iy = sparse.eye(ny, format='csc')
    op = sparse.kron(_axis(study, 'x'), iy, format='csc')
    op += sparse.kron(ix, _axis(study, 'y'), format='csc')
    identity = sparse.eye(nx * ny, format='csc')
    sources = np.asarray(study['source_modes'], dtype=float)
    rhs = sources.reshape(sources.shape[0], nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(op + float(mu) * identity).solve(rhs)
        blocks.append(fields[i * ny + j, :])
    return np.vstack(blocks), fields


def _profile(S, y, study):
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((S, y))
    gram = Z.T @ Z + float(study['nuisance_ridge']) * np.eye(Z.shape[1])
    residual = values - Z @ np.linalg.solve(gram, Z.T @ values)
    return residual[:, :-1], residual[:, -1]


def _qp(H, c, cap, budget):
    """Feasible active-set method; the equality multiplier is a gradient level."""
    n = len(c)
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = H @ w + c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    status = np.zeros(n, dtype=int)  # -1 low, +1 high, 0 free
    for _ in range(20 * n * n):
        g = H @ w + c
        tol = 2e-13 * max(1., np.max(abs(g)))
        free = np.flatnonzero(status == 0)
        if not free.size:
            low = np.flatnonzero(status == -1)
            high = np.flatnonzero(status == 1)
            i = low[np.argmin(g[low])]
            j = high[np.argmax(g[high])]
            if g[j] - g[i] <= tol:
                return w, float((g[i] + g[j]) / 2)
            status[[i, j]] = 0
            continue
        fixed = np.flatnonzero(status)
        nf = len(free)
        system = np.empty((nf + 1, nf + 1))
        system[:nf, :nf] = H[np.ix_(free, free)]
        system[nf, :nf] = 1.
        system[:nf, nf] = -1.
        system[nf, nf] = 0.
        rhs = np.r_[-c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
        sol = np.linalg.solve(system, rhs)
        target, nu = sol[:-1], float(sol[-1])
        p = target - w[free]
        ratios = np.full(nf, np.inf)
        lo = target < 0
        hi = target > cap
        ratios[lo] = -w[free[lo]] / p[lo]
        ratios[hi] = (cap - w[free[hi]]) / p[hi]
        j = np.argmin(ratios)
        if ratios[j] < 1:
            w[free] = np.clip(w[free] + ratios[j] * p, 0., cap)
            i = free[j]
            status[i] = -1 if lo[j] else 1
            w[i] = 0. if lo[j] else cap
            continue
        w[free] = np.clip(target, 0., cap)
        g = H @ w + c
        violation = (g - nu) * status
        j = np.argmax(violation)
        if violation[j] <= tol:
            return w, nu
        status[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_s00(study: dict) -> dict:
    """Reconstruct source weights and the terminal field from one study."""
    y = _calibrate(study)
    S, fields = _transport(study)
    S, y = _profile(S, y, study)
    ridge = float(study['source_ridge'])
    cap = float(study['u_box'])
    budget = float(study['total_budget'])
    H = S.T @ S + ridge * np.eye(S.shape[1])
    w, nu = _qp(H, -S.T @ y, cap, budget)
    r = S @ w - y
    g = S.T @ r + ridge * w
    primal = float((r @ r + ridge * (w @ w)) / 2)
    lower = np.maximum(g - nu, 0.)
    upper = np.maximum(nu - g, 0.)
    # Equivalent to the supporting-plane box infimum at feasible w.
    dual = primal - float(lower @ w + upper @ (cap - w))
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=lower, lam_upper=upper,
                primal_objective=primal, dual_lower_bound=dual)
    return dict(weights=w, terminal_field=(fields @ w).reshape(study['nx'], study['ny']),
                kkt_certificate=cert)
