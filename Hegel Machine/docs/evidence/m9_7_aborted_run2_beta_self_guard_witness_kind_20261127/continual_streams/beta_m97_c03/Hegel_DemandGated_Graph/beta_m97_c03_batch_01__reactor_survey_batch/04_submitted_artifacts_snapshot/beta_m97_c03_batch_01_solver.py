"""Bank-calibrated reaction–transport tomography with certified source inversion."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _clean_sweep(x, y):
    clean = np.ones(x.size, dtype=bool)
    if x.size >= 4:
        # Fit each one-point deletion. At four rows a single spike contaminates
        # half the pair slopes, but its deletion leaves three consistent rows.
        best_error = np.inf
        for i in range(x.size):
            keep = np.arange(x.size) != i
            if np.ptp(x[keep]) == 0:
                continue
            a, d = _affine(x[keep], y[keep])
            residual = y - (a*x + d)
            error = residual[keep] @ residual[keep]
            if error < best_error:
                best_error, omitted, best_residual = error, i, residual
        if np.isfinite(best_error):
            clean = np.abs(best_residual) <= 3.0
            if not clean[omitted] and clean.sum() == x.size - 1:
                return clean
    if x.size >= 3:
        # Use a robust pilot if no single deletion explains the contamination.
        ii, jj = np.triu_indices(x.size, 1)
        valid = x[ii] != x[jj]
        slopes = (y[jj[valid]] - y[ii[valid]]) / (x[jj[valid]] - x[ii[valid]])
        if slopes.size:
            a = np.median(slopes)
            d = np.median(y - a*x)
            return np.abs(y - (a*x + d)) <= 3.0
    return np.ones(x.size, dtype=bool)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    ref = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    gain_offset = np.empty((4, 2))
    history = None
    for b in range(4):
        x, y = ref[bank == b], raw[bank == b]
        if x.size >= 2:
            clean = _clean_sweep(x, y)
            if clean.all() or clean.sum() >= 4:
                gain_offset[b] = _affine(x[clean], y[clean])
                continue
        else:
            clean = np.ones(x.size, dtype=bool)
        if history is None and _ARCHIVE.is_file():
            history = json.loads(_ARCHIVE.read_text())
        if history is not None:
            for sweep in reversed(history):
                rows = np.asarray(sweep['bank_id']) == b
                hx = np.asarray(sweep['ref_signal'], dtype=float)[rows]
                hy = np.asarray(sweep['raw_readout'], dtype=float)[rows]
                usable = _clean_sweep(hx, hy)
                if usable.sum() >= 2 and np.ptp(hx[usable]) > 0:
                    gain_offset[b] = _affine(hx[usable], hy[usable])
                    break
            else:
                if clean.sum() < 2:
                    raise ValueError(f'Insufficient calibration data for bank {b}')
                gain_offset[b] = _affine(x[clean], y[clean])
            continue
        if clean.sum() >= 2:
            gain_offset[b] = _affine(x[clean], y[clean])
        else:
            raise ValueError(f'Insufficient calibration data for bank {b}')
    gain, offset = gain_offset[np.asarray(study['row_bank_id'], dtype=int)].T
    return (np.asarray(study['y_obs_raw'], dtype=float) - offset) / gain


def _axis(kap, vel, length, robin):
    kap, vel = np.asarray(kap), np.asarray(vel)
    h = length / (kap.size - 1)
    faces = (kap[:-1] + kap[1:]) / (2*h*h)
    lo = -faces - np.maximum(vel[1:], 0)/h
    hi = -faces - np.maximum(-vel[:-1], 0)/h
    diag = np.empty(kap.size)
    diag[1:-1] = faces[1:] + faces[:-1] + np.abs(vel[1:-1])/h
    hi[0] = -kap[0]/h**2
    lo[-1] = -kap[-1]/h**2
    diag[0], diag[-1] = -hi[0] + robin/h, -lo[-1] + robin/h
    return diags((lo, diag, hi), (-1, 0, 1), format='csc')


def _forward(study):
    nx, ny = study['nx'], study['ny']
    lx, ly = study['Lx'], study['Ly']
    trace = np.asarray(study['past_vortex_circulation_trace'])
    excess = np.maximum(0.0, trace - study['critical_vortex_circulation'])
    memory = float(excess @ excess)
    sealed = study['recirculation_loop_mode'] == 0 and memory > 0.0
    potential = np.asarray(study['V_reaction_2d'])
    if sealed:
        x = np.linspace(0.0, lx, nx)[:, None]
        y = np.linspace(0.0, ly, ny)[None, :]
        wall = np.exp(-(x/(0.25*lx))**2) + np.exp(-((lx-x)/(0.25*lx))**2)
        potential = potential + (0.95 + 0.55*np.tanh(memory))*wall*(0.6 + 0.4*np.sin(np.pi*y/ly))
    ax = _axis(study['kappa_x'], study['vel_x'], lx, 0.0 if sealed else study['robin_x'])
    ay = _axis(study['kappa_y'], study['vel_y'], ly, 0.0 if sealed else study['robin_y'])
    operator = kron(eye(ny), ax) + kron(ay, eye(nx)) + diags(potential.ravel(order='F'))
    ident = eye(nx*ny, format='csc')
    modes = np.asarray(study['source_modes']).transpose(0, 2, 1).reshape(-1, nx*ny).T
    i, j = np.asarray(study['sensor_ij']).T
    rows = i + nx*j
    blocks = []
    for mu in study['mu_shifts']:
        fields = splu((operator + mu*ident).tocsc()).solve(modes)
        blocks.append(fields[rows])
    return np.vstack(blocks), fields


def _invert(s, y, ridge, upper, budget):
    n = s.shape[1]
    gram = s.T @ s + ridge*np.eye(n)
    rhs = s.T @ y
    w = np.full(n, budget/n)
    # Keep primal feasibility while adding blocking bounds and release bounds with the wrong KKT sign.
    active = np.zeros(n, dtype=int)
    for _ in range(20*n*n):
        free = active == 0
        g = gram @ w - rhs
        if free.any():
            mat = gram[np.ix_(free, free)]
            inv_g, inv_one = np.linalg.solve(mat, np.column_stack((g[free], np.ones(free.sum())))).T
            nu = inv_g.sum()/inv_one.sum()
            direction = nu*inv_one - inv_g
            if np.max(np.abs(direction)) > 1e-12:
                ratio = np.full(direction.size, np.inf)
                up = direction > 1e-13
                down = direction < -1e-13
                wf = w[free]
                ratio[up] = (upper - wf[up])/direction[up]
                ratio[down] = -wf[down]/direction[down]
                block = np.argmin(ratio)
                step = min(1.0, max(0.0, ratio[block]))
                w[free] = np.clip(wf + step*direction, 0, upper)
                if step < 1.0:
                    idx = np.flatnonzero(free)[block]
                    active[idx] = 1 if up[block] else -1
                    w[idx] = upper if up[block] else 0.0
                continue
        else:
            lo = np.max(g[active == 1], initial=-np.inf)
            hi = np.min(g[active == -1], initial=np.inf)
            nu = np.clip(np.mean(g), min(lo, hi), max(lo, hi))
        violation = active*(g - nu)
        idx = np.argmax(violation)
        if violation[idx] <= 1e-11:
            break
        active[idx] = 0
    else:
        raise RuntimeError('Box-simplex active set did not converge')
    res = s @ w - y
    grad = s.T @ res + ridge*w
    slack = grad - nu
    obj = 0.5*(res @ res + ridge*(w @ w))
    dual = obj - grad @ w + nu*budget + upper*np.minimum(slack, 0).sum()
    return w, dict(weights=w.copy(), nu_sum=float(nu), lam_lower=np.maximum(slack, 0),
                   lam_upper=np.maximum(-slack, 0), primal_objective=float(obj),
                   dual_lower_bound=float(min(obj, dual)))



def solve_beta_m97_c03_batch_01(study: dict) -> dict:
    sensors, terminal = _forward(study)
    data = np.column_stack((_calibrate(study), sensors))
    z = np.asarray(study['Z_nuisance'])
    gram = z.T @ z + study['nuisance_ridge']*np.eye(z.shape[1])
    profiled = data - z @ np.linalg.solve(gram, z.T @ data)
    weights, certificate = _invert(profiled[:, 1:], profiled[:, 0], study['source_ridge'], study['u_box'], study['total_budget'])
    field = (terminal @ weights).reshape((study['nx'], study['ny']), order='F')
    return dict(weights=weights, terminal_field=field, kkt_certificate=certificate)
