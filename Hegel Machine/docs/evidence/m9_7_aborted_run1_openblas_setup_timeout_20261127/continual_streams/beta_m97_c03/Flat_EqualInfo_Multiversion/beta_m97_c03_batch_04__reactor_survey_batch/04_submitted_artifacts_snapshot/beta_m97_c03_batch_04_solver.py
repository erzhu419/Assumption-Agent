"""Calibrated reaction-transport tomography with sealed, Robin, and periodic boundaries."""
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


_CALIBRATION_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _fit_affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _clean_sweep(x, y):
    if x.size < 3:
        return np.ones(x.size, dtype=bool)
    # A repeated-median fit excludes isolated large excursions before least squares.
    dx = x[:, None] - x
    dy = y[:, None] - y
    slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
    gain = np.nanmedian(np.nanmedian(slopes, axis=1))
    offsets = y - gain * x
    clean = np.abs(offsets - np.median(offsets)) <= 3.0
    if x.size < 4 or not np.all(clean):
        return clean
    # A spike at a high-leverage reference can bias even the repeated
    # median just enough to mask an excursion near the cutoff. Check it
    # against a line fitted without that row; the retained sweep must
    # itself satisfy the affine law to calibration precision.
    order = np.argsort(x)
    for j in order:
        keep = order[order != j]
        lo, hi = keep[0], keep[-1]
        span = x[hi] - x[lo]
        if span == 0:
            continue
        t = (x - x[lo]) / span
        residual = (y - y[lo]) - (y[hi] - y[lo]) * t
        if np.max(np.abs(residual[keep])) > 1e-6:
            continue
        noise = 5e-7 * (1 + abs(1 - t[j]) + abs(t[j]))
        if abs(residual[j]) > 3.0 - noise:
            clean[j] = False
            break
    return clean


def _ambiguous_three_point_emi(x, y):
    if x.size != 3:
        return False
    # Any pair defines a line. Check collinearity before extrapolating
    # from a potentially close pair to test the 3-unit EMI threshold.
    lo, mid, hi = np.argsort(x)
    span = x[hi] - x[lo]
    if span == 0:
        return np.ptp(y) > 3.0
    deviation = abs((y[mid] - y[lo]) - (y[hi] - y[lo]) * (x[mid] - x[lo]) / span)
    if deviation <= 1e-5:  # Above the measured clean-sweep noise floor.
        return False
    short_span = min(x[mid] - x[lo], x[hi] - x[mid])
    return deviation * span > 3.0 * short_span - 1e-6 * span


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
        clean = _clean_sweep(x, y)
        prior = None
        if x.size < 2 or not np.all(clean) or _ambiguous_three_point_emi(x, y):
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
    for sweep in reversed(history):
        mask = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'], dtype=float)[mask]
        y = np.asarray(sweep['raw_readout'], dtype=float)[mask]
        clean = _clean_sweep(x, y)
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            return _fit_affine(x[clean], y[clean])
    return None


def _periodic_axis(kappa, vel, h):
    kappa, vel = np.asarray(kappa), np.asarray(vel)
    i = np.arange(kappa.size)
    prev, nxt = (i - 1) % i.size, (i + 1) % i.size
    face = (kappa + kappa[nxt]) / (2 * h**2)
    backward = face[prev] + np.maximum(vel, 0) / h
    forward = face + np.maximum(-vel, 0) / h
    return sparse.csc_matrix((np.r_[-backward, backward + forward, -forward],
                              (np.tile(i, 3), np.r_[prev, i, nxt])),
                             shape=(i.size, i.size))


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
    lx, ly = study['Lx'], study['Ly']
    hx, hy = lx / (nx - 1), ly / (ny - 1)
    mode = study['recirculation_loop_mode']
    potential = np.asarray(study['V_reaction_2d'])
    if mode > 0.50:
        ax = _periodic_axis(study['kappa_x'], study['vel_x'], hx)
        ay = _periodic_axis(study['kappa_y'], study['vel_y'], hy)
        cx = np.cos(2 * np.pi * np.linspace(0, lx, nx) / lx)
        cy = np.cos(2 * np.pi * np.linspace(0, ly, ny) / ly)
        potential = potential + 0.75 * mode * np.outer(cx, cy)
    else:
        rx, ry = study['robin_x'], study['robin_y']
        if mode == 0:
            excess = np.maximum(0.0, np.asarray(study['past_vortex_circulation_trace'])
                                - study['critical_vortex_circulation'])
            v_mem = float(excess @ excess)
            if v_mem > 0:
                rx = ry = 0.0  # Keep the one-sided diffusion rows, without Robin loss.
                xs, ys = np.linspace(0, lx, nx), np.linspace(0, ly, ny)
                wall = np.exp(-(xs / (0.25 * lx))**2) + np.exp(-((lx - xs) / (0.25 * lx))**2)
                profile = 0.6 + 0.4 * np.sin(np.pi * ys / ly)
                potential = potential + (0.95 + 0.55 * np.tanh(v_mem)) * np.outer(wall, profile)
        ax = _axis_operator(study['kappa_x'], study['vel_x'], rx, hx)
        ay = _axis_operator(study['kappa_y'], study['vel_y'], ry, hy)
    operator = (sparse.kron(sparse.eye(ny), ax, format='csc')
                + sparse.kron(ay, sparse.eye(nx), format='csc')
                + sparse.diags(potential.ravel(order='F'), format='csc'))
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


def solve_beta_m97_c03_batch_04(study: dict) -> dict:
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
