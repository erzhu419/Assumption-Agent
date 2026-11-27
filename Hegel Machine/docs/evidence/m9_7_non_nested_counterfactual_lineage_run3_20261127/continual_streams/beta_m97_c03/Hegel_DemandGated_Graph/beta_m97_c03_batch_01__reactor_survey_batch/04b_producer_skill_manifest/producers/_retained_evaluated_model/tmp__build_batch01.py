from pathlib import Path
archive = Path('/tmp/hegel_continual_archive/beta_m97_c03_s00_solver.py').read_text()
axis = archive[archive.index('def _axis('):archive.index('def _forward(')]
qp = archive[archive.index('def _box_simplex('):archive.index('def solve_beta_m97_c03_s00(')]
Path('/holdout_task/beta_m97_c03_batch_01_solver.py').write_text('''"""Calibrated reaction-transport inversion with a box-simplex certificate."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _fit_line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_sweep(x, y):
    """Keep sweep points within the EMI threshold of the consensus line."""
    if x.size < 3:
        return np.ones(x.size, dtype=bool)
    i, j = np.triu_indices(x.size, 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    intercepts = y[i] - slopes * x[i]
    error = np.abs(y[None, :] - slopes[:, None] * x[None, :] - intercepts[:, None])
    inlier = error <= 3.0
    count = inlier.sum(axis=1)
    best = np.lexsort((np.where(inlier, error**2, 0).sum(axis=1), -count))[0]
    return inlier[best]


def _archived_fit(bank):
    for sweep in reversed(json.loads(_HISTORY.read_text())):
        b = np.asarray(sweep['bank_id'])
        x = np.asarray(sweep['ref_signal'], dtype=float)[b == bank]
        y = np.asarray(sweep['raw_readout'], dtype=float)[b == bank]
        if np.unique(x).size >= 2:
            keep = _clean_sweep(x, y)
            return _fit_line(x[keep], y[keep])
    raise ValueError(f'No archived calibration for bank {bank}')


def _calibrate(study):
    sweep = study['sensor_calibration_sweeps']
    b = np.asarray(sweep['bank_id'])
    x = np.asarray(sweep['ref_signal'], dtype=float)
    y = np.asarray(sweep['raw_readout'], dtype=float)
    row_banks = np.asarray(study['row_bank_id'], dtype=int)
    coeff = np.empty((4, 2))
    for bank in np.unique(row_banks):
        xb, yb = x[b == bank], y[b == bank]
        keep = _clean_sweep(xb, yb)
        if xb.size < 2 or (not keep.all() and keep.sum() < 4):
            coeff[bank] = _archived_fit(bank)
        else:
            coeff[bank] = _fit_line(xb[keep], yb[keep])
    gain, offset = coeff[row_banks].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


''' + axis + '''def _forward(study):
    nx, ny = study['nx'], study['ny']
    lx, ly = study['Lx'], study['Ly']
    v = np.asarray(study['past_vortex_circulation_trace'])
    v_mem = np.sum(np.maximum(0.0, v - study['critical_vortex_circulation'])**2)
    sealed = study['recirculation_loop_mode'] == 0 and v_mem > 0
    ax, ay = [_axis(np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                    0.0 if sealed else study['robin_' + a], study['L' + a])
              for a in ('x', 'y')]
    reaction = np.asarray(study['V_reaction_2d'], dtype=float)
    if sealed:
        xs = np.linspace(0.0, lx, nx)[:, None]
        ys = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
        reaction = reaction + (0.95 + 0.55 * np.tanh(v_mem)) * wall * (0.6 + 0.4 * np.sin(np.pi * ys / ly))
    op = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc')
          + diags(reaction.ravel(order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    sensor_rows = i + nx * j
    identity = eye(nx * ny, format='csc')
    sampled = []
    for mu in study['mu_shifts']:
        field = splu(op + mu * identity).solve(modes)
        sampled.append(field[sensor_rows])
    return np.vstack(sampled), field


''' + qp + '''def solve_beta_m97_c03_batch_01(study: dict) -> dict:
    y = _calibrate(study)
    sensors, terminal = _forward(study)
    z = np.asarray(study['Z_nuisance'], dtype=float)
    data = np.column_stack((y, sensors))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    residualized = data - z @ np.linalg.solve(gram, z.T @ data)
    target, design = residualized[:, 0], residualized[:, 1:]
    ridge = float(study['source_ridge'])
    cap, budget = float(study['u_box']), float(study['total_budget'])
    hessian = design.T @ design + ridge * np.eye(design.shape[1])
    w, nu = _box_simplex(hessian, -design.T @ target, cap, budget)
    residual = design @ w - target
    gradient = design.T @ residual + ridge * w
    reduced = gradient - nu
    ll = np.where(w == 0.0, np.maximum(reduced, 0.0), 0.0)
    lu = np.where(w == cap, np.maximum(-reduced, 0.0), 0.0)
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    # Algebraically identical to the tangent lower bound, without cancellation
    # between the equality term and the full gradient inner product.
    gap = np.maximum(reduced, 0) @ w + np.maximum(-reduced, 0) @ (cap - w)
    certificate = {'weights': w.tolist(), 'nu_sum': nu, 'lam_lower': ll.tolist(),
                   'lam_upper': lu.tolist(), 'primal_objective': primal,
                   'dual_lower_bound': float(primal - gap)}
    field = (terminal @ w).reshape((study['nx'], study['ny']), order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': certificate}
''')
