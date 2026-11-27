"""Calibrated reaction-transport tomography with a box-simplex KKT certificate."""
from pathlib import Path
import json

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_calibration(x, y):
    # Estimate the clean line before applying the EMI threshold. Counting
    # points within 3 units first can reward a tilted line that hides a spike.
    best = None
    for i in range(x.size):
        for j in range(i):
            if x[i] == x[j]:
                continue
            slope = (y[i] - y[j]) / (x[i] - x[j])
            residual = np.abs(y - (y[j] + slope * (x - x[j])))
            mask = residual <= 3.0
            score = (-float(np.median(residual)), int(mask.sum()))
            if best is None or score > best[0]:
                best = score, mask
    return best[1] if best is not None else np.zeros(x.size, dtype=bool)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, ref, raw = (np.asarray(sweeps[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    coeffs = np.empty((4, 2))
    history = None
    for b in range(4):
        mask = banks == b
        x, y = ref[mask], raw[mask]
        clean = _clean_calibration(x, y)
        has_emi = np.unique(x).size >= 2 and not np.all(clean)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or not has_emi):
            coeffs[b] = _affine(x[clean], y[clean])
            continue
        if x.size < 2 or has_emi:
            if history is None:
                history = json.loads(_ARCHIVE.read_text())
            for record in reversed(history):
                mask = np.asarray(record['bank_id']) == b
                hx = np.asarray(record['ref_signal'])[mask]
                hy = np.asarray(record['raw_readout'])[mask]
                keep = _clean_calibration(hx, hy)
                if np.unique(hx[keep]).size >= 2:
                    coeffs[b] = _affine(hx[keep], hy[keep])
                    break
            else:
                raise ValueError(f'No usable archived calibration for bank {b}')
        else:
            raise ValueError(f'Bank {b} needs two distinct calibration references')
    gain, offset = coeffs[np.asarray(study['row_bank_id'])].T
    return (np.asarray(study['y_obs_raw']).reshape(-1) - offset) / gain


def _axis(kappa, velocity, robin, length):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    face = (kappa[:-1] + kappa[1:]) / (2 * h * h)
    diag = np.empty(n)
    diag[1:-1] = face[:-1] + face[1:] + np.abs(velocity[1:-1]) / h
    diag[[0, -1]] = kappa[[0, -1]] / (h * h) + robin / h
    lower = -face - np.maximum(velocity[1:], 0) / h
    upper = -face - np.maximum(-velocity[:-1], 0) / h
    lower[-1] = -kappa[-1] / (h * h)
    upper[0] = -kappa[0] / (h * h)
    return diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _periodic_axis(kappa, velocity, length):
    kappa, velocity = np.asarray(kappa), np.asarray(velocity)
    n = kappa.size
    h = length / (n - 1)
    i = np.arange(n)
    im, ip = (i - 1) % n, (i + 1) % n
    km = (kappa + kappa[im]) / (2 * h * h)
    kp = (kappa + kappa[ip]) / (2 * h * h)
    lower = -km - np.maximum(velocity, 0) / h
    upper = -kp - np.maximum(-velocity, 0) / h
    center = km + kp + np.abs(velocity) / h
    return coo_matrix((np.r_[lower, center, upper],
                       (np.tile(i, 3), np.r_[im, i, ip])), shape=(n, n)).tocsc()


def _forward(study):
    nx, ny = study['nx'], study['ny']
    lx, ly = study['Lx'], study['Ly']
    loop = study['recirculation_loop_mode']
    reaction = np.asarray(study['V_reaction_2d'])
    x = np.linspace(0.0, lx, nx)[:, None]
    y = np.linspace(0.0, ly, ny)[None, :]
    if loop > 0.50:
        ax = _periodic_axis(study['kappa_x'], study['vel_x'], lx)
        ay = _periodic_axis(study['kappa_y'], study['vel_y'], ly)
        reaction = reaction + 0.75 * loop * np.cos(2 * np.pi * x / lx) * np.cos(2 * np.pi * y / ly)
    else:
        excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                            - study['critical_vortex_circulation'])
        memory = float(excess @ excess)
        sealed = loop == 0 and memory > 0
        ax = _axis(study['kappa_x'], study['vel_x'], 0.0 if sealed else study['robin_x'], lx)
        ay = _axis(study['kappa_y'], study['vel_y'], 0.0 if sealed else study['robin_y'], ly)
        if sealed:
            wall = np.exp(-(x / (0.25 * lx))**2) + np.exp(-((lx - x) / (0.25 * lx))**2)
            reaction = reaction + (0.95 + 0.55 * np.tanh(memory)) * wall * (0.6 + 0.4 * np.sin(np.pi * y / ly))
    reaction = reaction.ravel(order='F')
    op = (kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(reaction)).tocsc()
    identity = eye(nx * ny, format='csc')
    rhs = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx * ny).T
    i, j = np.asarray(study['sensor_ij']).T
    blocks = []
    for shift in study['mu_shifts']:
        fields = splu(op + shift * identity).solve(rhs)
        blocks.append(fields[i + nx * j])
    return np.vstack(blocks), fields


def _box_simplex(H, c, cap, budget):
    """Solve strictly convex QP by exact minimization on successive faces."""
    n = c.size
    if budget == 0 or budget == n * cap:
        w = np.full(n, budget / n)
        g = H @ w - c
        return w, float(g.min() if budget == 0 else g.max())
    w = np.full(n, budget / n)
    side = np.zeros(n, dtype=int)  # -1 lower, +1 upper, 0 free
    for _ in range(100 * n):
        free = np.flatnonzero(side == 0)
        fixed = np.flatnonzero(side)
        if free.size:
            m = free.size
            system = np.ones((m + 1, m + 1))
            system[:m, :m] = H[np.ix_(free, free)]
            system[:m, m] = -1
            system[m, m] = 0
            rhs = np.r_[c[free] - H[np.ix_(free, fixed)] @ w[fixed], budget - w[fixed].sum()]
            sol = np.linalg.solve(system, rhs)
            target, nu = sol[:-1], float(sol[-1])
            step = target - w[free]
            crossings = np.full(m, np.inf)
            low = target < -1e-13
            high = target > cap + 1e-13
            crossings[low] = -w[free[low]] / step[low]
            crossings[high] = (cap - w[free[high]]) / step[high]
            hit = int(crossings.argmin())
            if crossings[hit] < 1:
                w[free] += max(0.0, crossings[hit]) * step
                side[free[hit]] = -1 if low[hit] else 1
                w[free[hit]] = 0 if low[hit] else cap
                continue
            w[free] = np.clip(target, 0, cap)
        else:
            g = H @ w - c
            low, high = np.flatnonzero(side < 0), np.flatnonzero(side > 0)
            i, j = low[g[low].argmin()], high[g[high].argmax()]
            if g[i] >= g[j] - 1e-12:
                return w, float((g[i] + g[j]) / 2)
            side[[i, j]] = 0
            continue
        g = H @ w - c
        violations = side * (g - nu)
        release = int(violations.argmax())
        if violations[release] <= 1e-12:
            return w, nu
        side[release] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_beta_m97_c01_batch_03(study: dict) -> dict:
    y = _calibrate(study)
    sensors, terminal_modes = _forward(study)
    z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((y, sensors))
    projected = data - z @ np.linalg.solve(z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ data)
    r, s = projected[:, 0], projected[:, 1:]
    ridge, cap, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(s.T @ s + ridge * np.eye(s.shape[1]), s.T @ r, cap, budget)
    error = s @ w - r
    g = s.T @ error + ridge * w
    primal = float(0.5 * (error @ error + ridge * (w @ w)))
    delta = g - nu
    dual = min(primal, primal - float(g @ w) + nu * budget + cap * float(np.minimum(delta, 0).sum()))
    return {
        'weights': w,
        'terminal_field': (terminal_modes @ w).reshape((study['nx'], study['ny']), order='F'),
        'kkt_certificate': {'weights': w.tolist(), 'nu_sum': nu,
                            'lam_lower': np.maximum(delta, 0).tolist(),
                            'lam_upper': np.maximum(-delta, 0).tolist(),
                            'primal_objective': primal, 'dual_lower_bound': dual},
    }
