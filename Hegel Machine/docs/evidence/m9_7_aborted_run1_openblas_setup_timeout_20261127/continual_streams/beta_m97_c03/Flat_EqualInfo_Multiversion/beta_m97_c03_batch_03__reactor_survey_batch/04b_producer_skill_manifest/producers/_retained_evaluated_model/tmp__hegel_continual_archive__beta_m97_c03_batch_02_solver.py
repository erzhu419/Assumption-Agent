"""Open-Robin reaction-transport tomography and box-simplex inversion."""
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _fit_affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gains = np.empty(4)
    offsets = np.empty(4)
    history = None
    for bank in range(4):
        mask = banks == bank
        x, y = ref[mask], raw[mask]
        spikes = np.zeros(x.size, dtype=bool)
        if x.size >= 3:
            # Repeated-median line keeps an isolated spike out of the reference fit.
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            offset = np.median(y - gain * x)
            spikes = np.abs(y - gain * x - offset) > 3.0
        clean = ~spikes
        prior = None
        if x.size < 2 or np.any(spikes):
            if history is None:
                try:
                    with open(_CALIBRATION_ARCHIVE) as f:
                        history = json.load(f)
                except FileNotFoundError:
                    history = {}
            prior = _archived_affine(history, bank)
        if clean.sum() >= 4 or (clean.sum() >= 2 and prior is None):
            gains[bank], offsets[bank] = _fit_affine(x[clean], y[clean])
        elif prior is not None:
            gains[bank], offsets[bank] = prior
        else:
            raise ValueError(f'Insufficient calibration data for bank {bank}')
    b = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw']) - offsets[b]) / gains[b]


def _archived_affine(history, bank):
    if isinstance(history, dict):
        if 'gain' in history and 'offset' in history:
            gain, offset = history['gain'], history['offset']
            if isinstance(gain, (list, dict)):
                return gain[str(bank) if isinstance(gain, dict) else bank], offset[str(bank) if isinstance(offset, dict) else bank]
            if history.get('bank_id') == bank:
                return gain, offset
        entry = history.get(str(bank))
        if isinstance(entry, dict) and 'gain' in entry and 'offset' in entry:
            return entry['gain'], entry['offset']
        records = history.get('banks', history.get('history', []))
        return _archived_affine(records, bank) if records else None
    for record in reversed(history):
        if isinstance(record, dict) and record.get('bank_id') == bank:
            return record['gain'], record['offset']
    return None


def _axis_operator(kap, vel, rob, h):
    kap, vel = np.asarray(kap), np.asarray(vel)
    n = kap.size
    face = (kap[1:] + kap[:-1]) / (2 * h**2)
    lower = -face - np.maximum(vel[1:], 0) / h
    upper = -face - np.maximum(-vel[:-1], 0) / h
    diag = np.empty(n)
    diag[1:-1] = face[:-1] + face[1:] + np.abs(vel[1:-1]) / h
    upper[0], lower[-1] = -kap[0] / h**2, -kap[-1] / h**2
    diag[0], diag[-1] = -upper[0] + rob / h, -lower[-1] + rob / h
    return sparse.diags((lower, diag, upper), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis_operator(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'] / (nx - 1))
    ay = _axis_operator(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'] / (ny - 1))
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    identity = sparse.eye(nx * ny, format='csc')
    modes = np.column_stack([mode.ravel(order='F') for mode in np.asarray(study['source_modes'])])
    i, j = np.asarray(study['sensor_ij']).T
    sensor_idx = i + nx * j
    sampled = []
    for mu in study['mu_shifts']:
        fields = splu(operator + mu * identity).solve(modes)
        sampled.append(fields[sensor_idx])
    return np.vstack(sampled), fields


def _box_simplex(S, y, ridge, upper, budget):
    """Solve the small QP by feasible moves to equality-constrained face minima."""
    m = S.shape[1]
    H = S.T @ S + ridge * np.eye(m)
    b = S.T @ y
    if budget == 0 or budget == m * upper:
        w = np.full(m, budget / m)
        g = H @ w - b
        return w, float(np.min(g) if budget == 0 else np.max(g))

    w = np.full(m, budget / m)
    active = np.zeros(m, dtype=int)  # -1 at zero, +1 at upper
    for _ in range(100 * m):
        free = np.flatnonzero(active == 0)
        if not free.size:
            g = H @ w - b
            lo = np.flatnonzero(active < 0)
            hi = np.flatnonzero(active > 0)
            enter, leave = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[enter] >= g[leave] - 1e-12:
                return w, float((g[enter] + g[leave]) / 2)
            active[[enter, leave]] = 0
            continue

        fixed = np.flatnonzero(active)
        rhs = b[free] - H[np.ix_(free, fixed)] @ w[fixed]
        hf = H[np.ix_(free, free)]
        base, unit = np.linalg.solve(hf, np.column_stack((rhs, np.ones(free.size)))).T
        nu = (budget - w[fixed].sum() - base.sum()) / unit.sum()
        target = base + nu * unit
        if np.any(target < -1e-13) or np.any(target > upper + 1e-13):
            move = target - w[free]
            a = np.full(free.size, np.inf)
            lower = target < -1e-13
            higher = target > upper + 1e-13
            a[lower] = -w[free[lower]] / move[lower]
            a[higher] = (upper - w[free[higher]]) / move[higher]
            hit = np.argmin(a)
            w[free] += a[hit] * move
            active[free[hit]] = -1 if lower[hit] else 1
            w[free[hit]] = 0.0 if lower[hit] else upper
            continue

        w[free] = np.clip(target, 0.0, upper)
        if not fixed.size:
            return w, float(nu)
        g = H @ w - b
        violation = active[fixed] * (g[fixed] - nu)
        release = np.argmax(violation)
        if violation[release] <= 1e-12:
            return w, float(nu)
        active[fixed[release]] = 0
    raise RuntimeError('Box-simplex active-set iteration did not converge')


def solve_beta_m97_c03_s00(study: dict) -> dict:
    S, fields = _forward(study)
    y = _calibrate(study)
    z = np.asarray(study['Z_nuisance'])
    data = np.column_stack((y, S))
    gram = z.T @ z + study['nuisance_ridge'] * np.eye(z.shape[1])
    projected = data - z @ np.linalg.solve(gram, z.T @ data)
    y, S = projected[:, 0], projected[:, 1:]
    ridge, upper, budget = (study[k] for k in ('source_ridge', 'u_box', 'total_budget'))
    w, nu = _box_simplex(S, y, ridge, upper, budget)
    residual = S @ w - y
    primal = float(0.5 * (residual @ residual + ridge * (w @ w)))
    g = S.T @ residual + ridge * w
    lower = primal - g @ w + nu * budget + upper * np.minimum(0.0, g - nu).sum()
    cert = dict(weights=w, nu_sum=nu, lam_lower=np.maximum(g - nu, 0),
                lam_upper=np.maximum(nu - g, 0), primal_objective=primal,
                dual_lower_bound=float(lower))
    terminal = (fields @ w).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=w, terminal_field=terminal, kkt_certificate=cert)
