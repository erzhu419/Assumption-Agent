"""Open-Robin reactor source inversion from calibrated sensor banks."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _bank_rows(sweeps, bank):
    mask = np.asarray(sweeps['bank_id']) == bank
    return (np.asarray(sweeps['ref_signal'], dtype=float)[mask],
            np.asarray(sweeps['raw_readout'], dtype=float)[mask])


def _fit_affine(x, y):
    xc = x - x.mean()
    gain = (xc @ (y - y.mean())) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _clean_rows(x, y):
    # Repeated-median slope keeps an isolated spike from setting the fit.
    dx = x[:, None] - x
    dy = y[:, None] - y
    slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
    gain = np.nanmedian(np.nanmedian(slopes, axis=1))
    intercept = np.median(y - gain * x)
    return np.abs(y - (gain * x + intercept)) <= 3.0


def _calibration(study):
    sweeps = study['sensor_calibration_sweeps']
    history = None
    gain = np.empty(4)
    offset = np.empty(4)
    for b in range(4):
        x, y = _bank_rows(sweeps, b)
        if len(x) >= 4:
            clean = _clean_rows(x, y)
            if clean.all() or clean.sum() >= 4:
                gain[b], offset[b] = _fit_affine(x[clean], y[clean])
                continue
        elif len(x) == 2:
            gain[b], offset[b] = _fit_affine(x, y)
            continue
        elif len(x) == 3:
            # An endpoint spike can be attenuated by interpolation. Test
            # every held-out row, then use history if the triplet is inconsistent.
            for j in range(3):
                i, k = (j + 1) % 3, (j + 2) % 3
                if x[i] != x[k]:
                    predicted = y[i] + (x[j] - x[i]) * (y[k] - y[i]) / (x[k] - x[i])
                    if abs(y[j] - predicted) > 3.0:
                        break
            else:
                gain[b], offset[b] = _fit_affine(x, y)
                continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        for sweep in reversed(history):
            hx, hy = _bank_rows(sweep, b)
            if len(hx) >= 4:
                clean = _clean_rows(hx, hy)
                hx, hy = hx[clean], hy[clean]
            if len(hx) >= 2 and np.ptp(hx) > 0:
                gain[b], offset[b] = _fit_affine(hx, hy)
                break
        else:
            raise ValueError(f'No usable archived calibration for sensor bank {b}')
    banks = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset[banks]) / gain[banks]


def _axis(kap, vel, robin, length):
    kap, vel = np.asarray(kap), np.asarray(vel)
    n = len(kap)
    h = length / (n - 1)
    face = (kap[:-1] + kap[1:]) / (2 * h**2)
    low = -face - np.maximum(vel[1:], 0) / h
    up = -face - np.maximum(-vel[:-1], 0) / h
    main = np.empty(n)
    main[1:-1] = face[:-1] + face[1:] + np.abs(vel[1:-1]) / h
    main[[0, -1]] = kap[[0, -1]] / h**2 + robin / h
    up[0], low[-1] = -kap[0] / h**2, -kap[-1] / h**2
    return diags((low, main, up), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    ax = _axis(study['kappa_x'], study['vel_x'], study['robin_x'], study['Lx'])
    ay = _axis(study['kappa_y'], study['vel_y'], study['robin_y'], study['Ly'])
    L = (kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') +
         diags(np.asarray(study['V_reaction_2d']).ravel(order='F'), format='csc'))
    modes = np.asarray(study['source_modes']).reshape(-1, nx * ny, order='F').T
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    sensor = i + nx * j
    I = eye(nx * ny, format='csc')
    blocks = []
    for mu in study['mu_shifts']:
        response = splu(L + mu * I).solve(modes)
        blocks.append(response[sensor])
    return np.concatenate(blocks), response


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    # -1 marks a lower bound, +1 an upper bound, and 0 a free coordinate.
    active = np.zeros(n, dtype=int)
    w = np.full(n, budget / n)
    for _ in range(20 * n * n):
        g = H @ w - c
        if budget == 0 or cap == 0:
            nu = g.min()
            break
        if budget == n * cap:
            nu = g.max()
            break
        free = np.flatnonzero(active == 0)
        lower = np.flatnonzero(active < 0)
        upper = np.flatnonzero(active > 0)
        if not len(free):
            lo = lower[np.argmin(g[lower])]
            hi = upper[np.argmax(g[upper])]
            if g[hi] <= g[lo] + 1e-12:
                nu = (g[hi] + g[lo]) / 2
                break
            active[[lo, hi]] = 0
            continue

        hf = H[np.ix_(free, free)]
        rhs = c[free] - cap * H[np.ix_(free, upper)].sum(axis=1)
        z, v = np.linalg.solve(hf, np.column_stack((rhs, np.ones(len(free))))).T
        nu = (budget - cap * len(upper) - z.sum()) / v.sum()
        target = z + nu * v
        step = target - w[free]
        below, above = target < -1e-13, target > cap + 1e-13
        if below.any() or above.any():
            alpha = np.full(len(free), np.inf)
            alpha[below] = -w[free[below]] / step[below]
            alpha[above] = (cap - w[free[above]]) / step[above]
            hit = int(np.argmin(alpha))
            w[free] = np.clip(w[free] + max(0.0, alpha[hit]) * step, 0, cap)
            active[free[hit]] = 1 if above[hit] else -1
            w[free[hit]] = cap if above[hit] else 0
            continue

        w[free] = np.clip(target, 0, cap)
        g = H @ w - c
        violation = active * (g - nu)
        worst = int(np.argmax(violation))
        if violation[worst] <= 1e-12:
            break
        active[worst] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')

    residual = S @ w - y
    primal = 0.5 * (residual @ residual + ridge * (w @ w))
    g = S.T @ residual + ridge * w
    shifted = g - nu
    lower = np.where(w == 0, np.maximum(shifted, 0), 0)
    upper = np.where(w == cap, np.maximum(-shifted, 0), 0)
    dual = primal - g @ w + nu * budget + cap * np.minimum(shifted, 0).sum()
    return w, {'weights': w.copy(), 'nu_sum': float(nu), 'lam_lower': lower,
               'lam_upper': upper, 'primal_objective': float(primal),
               'dual_lower_bound': float(min(primal, dual))}


def solve_beta_m97_c02_batch_02(study: dict) -> dict:
    y = _calibration(study)
    S, terminal = _forward(study)
    Z = np.asarray(study['Z_nuisance'])
    values = np.column_stack((y, S))
    gram = Z.T @ Z + study['nuisance_ridge'] * np.eye(Z.shape[1])
    residual = values - Z @ np.linalg.solve(gram, Z.T @ values)
    w, cert = _box_simplex(residual[:, 1:], residual[:, 0], study['source_ridge'],
                           study['u_box'], study['total_budget'])
    field = (terminal @ w).reshape(study['nx'], study['ny'], order='F')
    return {'weights': w, 'terminal_field': field, 'kkt_certificate': cert}
