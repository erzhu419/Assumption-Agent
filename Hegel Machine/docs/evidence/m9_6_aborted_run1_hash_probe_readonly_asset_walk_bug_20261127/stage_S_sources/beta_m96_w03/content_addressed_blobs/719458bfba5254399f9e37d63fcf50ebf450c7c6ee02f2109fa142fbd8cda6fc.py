"""Calibrated tracer tomography for the laminar reaction-transport model."""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    k = np.asarray(kappa, dtype=float)
    v = np.asarray(vel, dtype=float) / h
    face = (2 * k[:-1] * k[1:] / (k[:-1] + k[1:])) / h**2
    diag = np.array(diag_add, dtype=float, copy=True) + np.abs(v)
    diag[:-1] += face
    diag[1:] += face
    kb = k[[0, -1]]
    wall = robin * kb / (h * (robin * h / 2 + kb))
    diag[0] += wall[0]
    diag[-1] += wall[1]
    return diag, -face - np.maximum(v[1:], 0), -face + np.minimum(v[:-1], 0)


def _axis_modes(diag, lower, upper):
    # A = D Q diag(eigenvalues) Q.T D^-1 for the upwind tridiagonal A.
    log_scale = np.r_[0., np.cumsum(0.5 * np.log(lower / upper))]
    scale = np.exp(log_scale - np.mean(log_scale))
    eig, basis = eigh_tridiagonal(diag, -np.sqrt(lower * upper))
    return eig, scale[:, None] * basis, basis.T / scale


def _transport(s):
    v = np.asarray(s['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    if (not np.allclose(v, vx[:, None] + vy, rtol=1e-10, atol=1e-12)
            or np.max(s['past_vortex_circulation_trace']) > s['critical_vortex_circulation']):
        raise ValueError('study is outside the laminar separable regime')
    ex, X, Xi = _axis_modes(*_build_1d(s['Lx'], s['nx'], s['kappa_x'], s['vel_x'],
                                       s['robin_x'], np.asarray(s['reaction_x']) + vx))
    ey, Y, Yi = _axis_modes(*_build_1d(s['Ly'], s['ny'], s['kappa_y'], s['vel_y'],
                                       s['robin_y'], np.asarray(s['reaction_y']) + vy))
    rhs = (Xi @ np.asarray(s['source_modes'], dtype=float)) @ Yi.T
    eigen_sum = ex[:, None] + ey
    i, j = np.asarray(s['sensor_ij'], dtype=int).T
    blocks = []
    for mu in s['mu_shifts']:
        spectral = rhs / (eigen_sum + mu)
        # Evaluate only paired sensor coordinates at intermediate shifts.
        blocks.append(np.einsum('msy,sy->sm', X[i] @ spectral, Y[j]))
    return np.concatenate(blocks, axis=0), (X @ spectral) @ Y.T


def _fit_bank(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError('bank calibration needs two distinct reference signals')
    use = np.ones(len(x), dtype=bool)
    if len(x) >= 4:
        # Use a high-breakdown pilot to keep isolated EMI spikes out of the refit.
        i, j = np.triu_indices(len(x), 1)
        good = abs(x[j] - x[i]) > 1e-12 * np.ptp(x)
        slope = np.median((y[j[good]] - y[i[good]]) / (x[j[good]] - x[i[good]]))
        deviation = y - slope * x
        dev = abs(deviation - np.median(deviation))
        inliers = dev <= max(6 * 1.4826 * np.median(dev), 1e-12 * max(1., np.max(abs(y))))
        if inliers.sum() >= 2 and np.ptp(x[inliers]) > 0:
            use = inliers
    return np.linalg.lstsq(np.column_stack((x[use], np.ones(use.sum()))), y[use], rcond=None)[0]


def _observations(study):
    sweep = study['sensor_calibration_sweeps']
    banks = np.asarray(sweep['bank_id'], dtype=int)
    x, y = (np.asarray(sweep[k], dtype=float) for k in ('ref_signal', 'raw_readout'))
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    corrected = np.asarray(study['y_obs_raw'], dtype=float).copy()
    for bank in np.unique(row_banks):
        gain, offset = _fit_bank(x[banks == bank], y[banks == bank])
        rows = row_banks == bank
        corrected[rows] = (corrected[rows] - offset) / gain
    return corrected


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    w = np.full(n, budget / n)
    g = H @ w - c
    if budget == 0 or budget == n * cap:
        return w, float(np.min(g) if budget == 0 else np.max(g))
    gradient_tol = 32 * np.finfo(float).eps * max(1., np.linalg.norm(H, np.inf) * cap + np.linalg.norm(c, np.inf))
    status = np.zeros(n, dtype=int)  # -1: lower bound; +1: upper bound
    for _ in range(20 * n * n):
        free = np.flatnonzero(status == 0)
        g = H @ w - c
        if not free.size:
            lo, hi = np.flatnonzero(status < 0), np.flatnonzero(status > 0)
            enter, leave = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[enter] >= g[leave] - gradient_tol:
                return w, float((g[enter] + g[leave]) / 2)
            status[[enter, leave]] = 0
            continue
        A = H[np.ix_(free, free)]
        ones = np.ones(free.size)
        a, b = np.linalg.solve(A, np.column_stack((g[free], ones))).T
        nu = float(a.sum() / b.sum())
        step = nu * b - a
        # One Newton step minimizes the quadratic on this face. If it
        # reaches the face minimum, test multipliers rather than re-solving
        # the same ill-conditioned system to chase roundoff in the step.
        trial = w[free] + step
        tol = 32 * np.finfo(float).eps * cap
        up, down = trial > cap + tol, trial < -tol
        if np.any(up | down):
            ratios = np.full(free.size, np.inf)
            ratios[up] = (cap - w[free[up]]) / step[up]
            ratios[down] = -w[free[down]] / step[down]
            hit = int(np.argmin(ratios))
            w[free] = np.clip(w[free] + np.clip(ratios[hit], 0., 1.) * step, 0., cap)
            k = free[hit]
            status[k] = 1 if up[hit] else -1
            w[k] = cap if up[hit] else 0.
            continue
        w[free] = np.clip(trial, 0., cap)
        g = H @ w - c
        nu = float(np.mean(g[free]))
        violation = status * (g - nu)
        k = int(np.argmax(violation))
        if violation[k] <= gradient_tol:
            return w, nu
        status[k] = 0
    raise RuntimeError('box-simplex active set did not converge')


def solve_beta_m96_w03_s00(study: dict) -> dict:
    design, terminal_modes = _transport(study)
    Z = np.asarray(study['Z_nuisance'], dtype=float)
    values = np.column_stack((design, _observations(study)))
    rho = float(study['nuisance_ridge'])
    if rho > 0:
        coef = np.linalg.solve(Z.T @ Z + rho * np.eye(Z.shape[1]), Z.T @ values)
    else:
        coef = np.linalg.lstsq(Z, values, rcond=None)[0]
    profiled = values - Z @ coef
    S, y = profiled[:, :-1], profiled[:, -1]
    ridge, cap, budget = (float(study[key]) for key in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, y, ridge, cap, budget)
    residual = S @ w - y
    gradient = S.T @ residual + ridge * w
    p = 0.5 * (residual @ residual + ridge * (w @ w))
    lower = np.maximum(gradient - nu, 0.)
    upper = np.maximum(nu - gradient, 0.)
    d = p - (lower @ w + upper @ (cap - w)) + nu * (budget - w.sum())
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=lower, lam_upper=upper,
                primal_objective=float(p), dual_lower_bound=float(d))
    return dict(weights=w, terminal_field=np.einsum('m,mij->ij', w, terminal_modes),
                kkt_certificate=cert)
