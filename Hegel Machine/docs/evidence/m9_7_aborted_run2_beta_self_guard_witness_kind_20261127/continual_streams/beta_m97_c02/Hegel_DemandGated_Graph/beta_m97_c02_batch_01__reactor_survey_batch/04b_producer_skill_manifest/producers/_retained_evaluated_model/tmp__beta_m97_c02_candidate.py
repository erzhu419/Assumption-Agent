"""Calibrated reaction–transport tomography and certified box-simplex source inversion."""

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


def _calibration_parameters(sweeps):
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    parameters = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        candidates = None
        if x.size >= 2 and np.ptp(x) > 0:
            gain, offset = _line(x, y)
            if x.size >= 3 and np.max(np.abs(y - (gain * x + offset))) > 1e-5:
                # Clean calibration rows agree with a line to measurement precision.
                # Keep only hypotheses that also explain rejected rows as EMI.
                i, j = np.triu_indices(x.size, k=1)
                distinct = x[i] != x[j]
                i, j = i[distinct], j[distinct]
                slopes = (y[j] - y[i]) / (x[j] - x[i])
                offsets = y[i] - slopes * x[i]
                residual = np.abs(y - (slopes[:, None] * x + offsets[:, None]))
                inlier = residual <= 1e-5
                valid = ((inlier | (residual > 3.0)).all(axis=1)
                         & (residual > 3.0).any(axis=1))
                if valid.any():
                    support = inlier.sum(axis=1)
                    candidates = np.unique(inlier[valid & (support == support[valid].max())], axis=0)
            if candidates is None:
                parameters[b] = gain, offset
                continue
            if len(candidates) == 1 or candidates[0].sum() >= 4:
                # Prefer the tightest fit if equally supported clean subsets exist.
                fits = [_line(x[m], y[m]) for m in candidates]
                parameters[b] = min(zip(fits, candidates), key=lambda fm: np.linalg.norm(
                    y[fm[1]] - (fm[0][0] * x[fm[1]] + fm[0][1])))[0]
                continue
        elif x.size >= 2:
            raise ValueError(f'Bank {b} sweeps have no distinct reference values')
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        px = np.concatenate([np.asarray(row['ref_signal'], dtype=float)[np.asarray(row['bank_id']) == b]
                             for row in history])
        py = np.concatenate([np.asarray(row['raw_readout'], dtype=float)[np.asarray(row['bank_id']) == b]
                             for row in history])
        prior = np.array(_line(px, py))
        if candidates is None:
            parameters[b] = prior
        else:
            # Prefix calibration breaks the tie; the retained rows determine the new line.
            fits = np.array([_line(x[m], y[m]) for m in candidates])
            parameters[b] = fits[np.argmin(np.sum((fits - prior)**2, axis=1))]
    return parameters


def _calibrate(study):
    parameters = _calibration_parameters(study['sensor_calibration_sweeps'])
    gain, offset = parameters[np.asarray(study['row_bank_id'], dtype=int)].T
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
    lx, ly = study['Lx'], study['Ly']
    excursion = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                           - study['critical_vortex_circulation'])
    memory = np.sum(excursion**2)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0
    ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx / (nx - 1))
    # Fouling seals both axes; the sorption sink is localized at the x walls.
    ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], ly / (ny - 1))
    reaction = np.asarray(study['V_reaction_2d'])
    if sealed:
        xs = np.linspace(0, lx, nx)[:, None]
        ys = np.linspace(0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(memory)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    operator = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
    operator += diags(reaction.ravel(order='F'), format='csc')
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


def solve_beta_m97_c02_batch_01(study: dict) -> dict:
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
