"""History-gated tracer tomography with affine calibration and a box-simplex certificate."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _samples(sweeps, bank):
    take = np.asarray(sweeps['bank_id']) == bank
    return tuple(np.asarray(sweeps[key], dtype=float)[take]
                 for key in ('ref_signal', 'raw_readout'))


def _current_line(x, y, prior=None):
    # The archived sweep coordinates are rounded to six decimal places.
    tol = 2e-6
    if x.size >= 3 and np.ptp(x) > 0:
        gain, offset = _line(x, y)
        if gain > 0 and np.max(np.abs(y - (gain * x + offset))) <= tol:
            return gain, offset
    best = None
    for i in range(x.size):
        for j in range(i):
            if abs(x[i] - x[j]) <= tol:
                continue
            gain = (y[i] - y[j]) / (x[i] - x[j])
            if gain <= 0:
                continue
            offset = y[i] - gain * x[i]
            residual = np.abs(y - (gain * x + offset))
            keep = residual <= tol
            count = int(keep.sum())
            # Two checks do not give an independent consistency check on a fitted line.
            if count < max(3, x.size // 2 + 1):
                continue
            gain, offset = _line(x[keep], y[keep])
            distance = 0.0 if prior is None else (gain - prior[0])**2 + (offset - prior[1])**2
            key = (-count, distance, float(np.max(residual[keep])))
            if best is None or key < best[0]:
                best = key, (gain, offset)
    return prior if best is None else best[1]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    row_bank = np.asarray(study['row_bank_id'])
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    calibrated = np.empty_like(raw)
    history = None
    for bank in np.unique(row_bank):
        x, y = _samples(sweeps, bank)
        line = _current_line(x, y)
        if line is None:
            if history is None:
                history = json.loads(_CALIBRATION_HISTORY.read_text())
            for episode in reversed(history):
                prior = _current_line(*_samples(episode, bank))
                if prior is not None:
                    line = _current_line(x, y, prior)
                    break
        if line is None:
            raise ValueError(f'Bank {bank} has no identifiable calibration line')
        gain, offset = line
        take = row_bank == bank
        calibrated[take] = (raw[take] - offset) / gain
    return calibrated


def _build_1d(L, n, kappa, vel, robin, diag_add):
    k = np.asarray(kappa, dtype=float)
    h = L / n
    faces = 2 * k[:-1] * k[1:] / ((k[:-1] + k[1:]) * h**2)
    positive = np.maximum(vel, 0) / h
    negative = np.maximum(-np.asarray(vel), 0) / h
    diag = np.asarray(diag_add) + np.r_[faces, 0.] + np.r_[0., faces] + positive + negative
    for i in (0, n - 1):
        diag[i] += robin * k[i] / (h * (0.5 * robin * h + k[i]))
    return np.diag(diag) - np.diag(faces + negative[:-1], 1) - np.diag(faces + positive[1:], -1)


def _diffusivity_factor(study, v):
    interaction = v - v.mean(axis=0) - v.mean(axis=1, keepdims=True) + v.mean()
    norm = np.linalg.norm(v)
    nonseparable = norm > 0 and np.linalg.norm(interaction) / norm > 1e-4
    memory = np.max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']
    return 1.45 if nonseparable or memory else 1.0


def _forward(study):
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    factor = _diffusivity_factor(study, v)
    nx, ny = v.shape
    axes = [csc_matrix(_build_1d(study['L' + a], study['n' + a],
                                factor * np.asarray(study['kappa_' + a]),
                                study['vel_' + a], study['robin_' + a],
                                study['reaction_' + a])) for a in 'xy']
    ident = eye(nx * ny, format='csc')
    operator = (kron(axes[0], eye(ny), format='csc') +
                kron(eye(nx), axes[1], format='csc') + diags(v.ravel(), format='csc'))
    src = np.asarray(study['source_modes'], dtype=float)
    rhs = src.reshape(src.shape[0], -1).T
    si, sj = np.asarray(study['sensor_ij']).T
    sample = si * ny + sj
    blocks = []
    for mu in study['mu_shifts']:
        states = splu(operator + float(mu) * ident).solve(rhs)
        blocks.append(states[sample])
    return np.concatenate(blocks, axis=0), states.T.reshape(src.shape)


def _residualize(z, ridge, columns):
    # Apply the ridge residual maker to the design and target, not its square root.
    q, d, _ = np.linalg.svd(z, full_matrices=False)
    fraction = np.divide(d*d, d*d + ridge, out=np.zeros_like(d), where=(d*d + ridge) > 0)
    return columns - q @ (fraction[:, None] * (q.T @ columns))


def _fit_weights(s, y, ridge, cap, budget):
    m = s.shape[1]
    h = s.T @ s + ridge * np.eye(m)
    c = s.T @ y
    w = np.full(m, budget / m)
    if budget == 0 or budget == cap * m:
        g = h @ w - c
        return w, float(g.min() if budget == 0 else g.max())

    # Optimize exactly on the current face. Block the step at the first bound;
    # release a bound only if its multiplier has the wrong sign.
    active = np.zeros(m, dtype=int)
    tol = 1e-12 * max(1., np.linalg.norm(h, ord=np.inf) * cap, np.max(np.abs(c)))
    for _ in range(100 * m):
        free = np.flatnonzero(active == 0)
        g = h @ w - c
        if free.size:
            hf = h[np.ix_(free, free)]
            inv_g, inv_one = np.linalg.solve(hf, np.column_stack((g[free], np.ones(free.size)))).T
            nu = inv_g.sum() / inv_one.sum()
            step = nu * inv_one - inv_g
            ratios = np.full(free.size, np.inf)
            down, up = step < 0, step > 0
            ratios[down] = -w[free[down]] / step[down]
            ratios[up] = (cap - w[free[up]]) / step[up]
            hit = int(ratios.argmin())
            alpha = min(1., max(0., ratios[hit]))
            w[free] += alpha * step
            if alpha < 1.:
                j = free[hit]
                active[j] = -1 if step[hit] < 0 else 1
                w[j] = 0. if active[j] == -1 else cap
                continue
            g = h @ w - c
            nu = g[free].mean()
        else:
            lo, hi = g[active == 1].max(), g[active == -1].min()
            nu = 0.5 * (lo + hi)
        violation = active * (g - nu)
        j = int(violation.argmax())
        if violation[j] <= tol:
            return w, float(nu)
        active[j] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m96_w01_batch_04(study: dict) -> dict:
    y = _calibrate(study)
    design, last = _forward(study)
    columns = _residualize(study['Z_nuisance'], study['nuisance_ridge'], np.column_stack((design, y)))
    s, y = columns[:, :-1], columns[:, -1]
    ridge, cap, budget = (float(study[k]) for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _fit_weights(s, y, ridge, cap, budget)
    res = s @ w - y
    g = s.T @ res + ridge * w
    objective = float(0.5 * (res @ res + ridge * (w @ w)))
    dual = objective - g @ w + nu * budget + cap * np.minimum(0., g - nu).sum()
    cert = dict(weights=w.copy(), nu_sum=nu, lam_lower=np.maximum(0., g - nu),
                lam_upper=np.maximum(0., nu - g), primal_objective=objective,
                dual_lower_bound=min(objective, float(dual)))
    return dict(weights=w, terminal_field=np.einsum('m,mij->ij', w, last), kkt_certificate=cert)
