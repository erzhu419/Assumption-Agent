"""Open-Robin reaction–transport tomography and certified source inversion."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    parameters = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        clean = np.ones(x.size, dtype=bool)
        if x.size >= 3:
            i, j = np.triu_indices(x.size, k=1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            if i.size:
                # Median of pointwise median slopes, with centered intercept.
                slopes = (y[j] - y[i]) / (x[j] - x[i])
                medians = [np.median(slopes[(i == k) | (j == k)]) for k in range(x.size)]
                gain = np.median(medians)
                offset = np.median(y - gain * x)
                clean = np.abs(y - (gain * x + offset)) <= 3.0
        if x.size < 2 or not clean.all():
            if history is None:
                try:
                    history = json.loads(_ARCHIVE.read_text())
                except FileNotFoundError:
                    history = {}
        if np.ptp(x[clean]) > 0 if clean.sum() >= 2 else False:
            parameters[b] = _line(x[clean], y[clean])
        else:
            prior = (history or {}).get(str(b))
            if prior is None:
                raise ValueError(f'Bank {b} has no identifiable in-batch calibration or archived calibration')
            parameters[b] = prior['gain'], prior['offset']
    gain, offset = parameters[np.asarray(study['row_bank_id'])].T
    return (np.asarray(study['y_obs_raw']) - offset) / gain


def _axis(kappa, velocity, robin, h):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    face = (kappa[:-1] + kappa[1:]) / (2 * h**2)
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    main = np.empty(kappa.size)
    main[1:-1] = face[:-1] + face[1:] + np.abs(velocity[1:-1]) / h
    upper[0], lower[-1] = -kappa[0] / h**2, -kappa[-1] / h**2
    main[0], main[-1] = -upper[0] + robin / h, -lower[-1] + robin / h
    return diags((lower, main, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'] / (ny - 1))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    operator += diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc')
    identity = eye(nx * ny, format='csc')
    sources = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    ix, iy = np.asarray(study['sensor_ij']).T
    rows = ix + nx * iy
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(sources)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields


def _box_simplex(hessian, linear, cap, budget):
    """Primal active set; the equality multiplier uses the gradient sign."""
    n = linear.size
    w = np.full(n, budget / n)
    if budget == 0 or budget == n * cap:
        g = hessian @ w - linear
        return w, float(g.min() if budget == 0 else g.max())
    active = np.zeros(n, dtype=int)
    for _ in range(20 * n**2):
        free = np.flatnonzero(active == 0)
        g = hessian @ w - linear
        if not free.size:
            lo = np.flatnonzero(active < 0)
            hi = np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue
        k = free.size
        kkt = np.ones((k + 1, k + 1))
        kkt[:k, :k] = hessian[np.ix_(free, free)]
        kkt[:k, k] = -1
        kkt[k, k] = 0
        p = np.flatnonzero(active != 0)
        sol = np.linalg.solve(kkt, np.r_[linear[free] - hessian[np.ix_(free, p)] @ w[p], budget - w[p].sum()])
        trial, nu = sol[:-1], float(sol[-1])
        direction = trial - w[free]
        steps = np.full(k, np.inf)
        lower = trial < -1e-13
        upper = trial > cap + 1e-13
        steps[lower] = -w[free[lower]] / direction[lower]
        steps[upper] = (cap - w[free[upper]]) / direction[upper]
        hit = np.argmin(steps)
        if steps[hit] < 1:
            w[free] += steps[hit] * direction
            i = free[hit]
            active[i], w[i] = (-1, 0.0) if lower[hit] else (1, cap)
            continue
        w[free] = np.clip(trial, 0, cap)
        g = hessian @ w - linear
        violation = active * (g - nu)
        i = np.argmax(violation)
        if violation[i] <= 1e-12:
            return w, nu
        active[i] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c02_s00(study: dict) -> dict:
    raw_design, terminal_basis = _forward(study)
    z = np.asarray(study['Z_nuisance'])
    stacked = np.column_stack((_calibrate(study), raw_design))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    residualized = stacked - z @ np.linalg.solve(gram, z.T @ stacked)
    target, design = residualized[:, 0], residualized[:, 1:]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ target, cap, budget)
    residual = design @ w - target
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = design.T @ residual + ridge * w
    dual = objective - g @ w + nu * budget + cap * np.minimum(0, g - nu).sum()
    certificate = dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(0, g - nu),
                       lam_upper=np.maximum(0, nu - g), primal_objective=objective,
                       dual_lower_bound=float(dual))
    return dict(weights=w, terminal_field=(terminal_basis @ w).reshape((study['nx'], study['ny']), order='F'),
                kkt_certificate=certificate)
