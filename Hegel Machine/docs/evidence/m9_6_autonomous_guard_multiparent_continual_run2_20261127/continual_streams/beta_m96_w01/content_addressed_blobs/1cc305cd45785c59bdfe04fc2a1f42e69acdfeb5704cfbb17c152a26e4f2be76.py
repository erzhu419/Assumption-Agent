"""Calibrated multi-wing tracer tomography with circulation memory."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _prior_bank(bank):
    with _HISTORY.open() as handle:
        history = json.load(handle)
    for sweep in reversed(history):
        mask = np.asarray(sweep['bank_id']) == bank
        x = np.asarray(sweep['ref_signal'])[mask]
        y = np.asarray(sweep['raw_readout'])[mask]
        if x.size >= 2 and np.ptp(x) > 0:
            return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
    raise ValueError(f'No identifiable calibration for bank {bank}')


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    bank = np.asarray(sweeps['bank_id'])
    refs = np.asarray(sweeps['ref_signal'], dtype=float)
    raw = np.asarray(sweeps['raw_readout'], dtype=float)
    row_bank = np.asarray(study['row_bank_id'])
    y = np.asarray(study['y_obs_raw'], dtype=float)
    calibrated = np.empty_like(y)
    for b in np.unique(row_bank):
        x, z = refs[bank == b], raw[bank == b]
        if x.size < 2 or np.ptp(x) == 0:
            gain, offset = _prior_bank(b)
            rows = row_bank == b
            calibrated[rows] = (y[rows] - offset) / gain
            continue
        if x.size == 4:
            # Three agreeing reference points can identify one isolated spike.
            candidates = []
            for i in range(4):
                keep = np.arange(4) != i
                a, v = x[keep], z[keep]
                ac = a - a.mean()
                if np.ptp(a) == 0:
                    continue
                slope = ac @ (v - v.mean()) / (ac @ ac)
                train_error = np.max(np.abs(v - v.mean() - slope * ac))
                held_error = abs(z[i] - v.mean() - slope * (x[i] - a.mean()))
                candidates.append((train_error, -held_error, i))
            error, neg_held_error, i = min(candidates)
            floor = 32 * np.finfo(float).eps * max(1., np.max(np.abs(z)))
            if -neg_held_error > 8 * max(error, floor):
                keep = np.arange(4) != i
                x, z = x[keep], z[keep]
        elif x.size >= 5:
            # A repeated-median line resists isolated bad readings, even at levered references.
            dx = x[:, None] - x
            dz = z[:, None] - z
            slopes = np.divide(dz, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            slope = np.median(np.nanmedian(slopes, axis=1))
            err = z - slope * x
            dev = np.abs(err - np.median(err))
            cutoff = max(6 * 1.4826 * np.median(dev),
                         32 * np.finfo(float).eps * max(1., np.max(np.abs(z))))
            keep = dev <= cutoff
            if np.any(keep) and np.ptp(x[keep]) > 0:
                x, z = x[keep], z[keep]
        xc = x - x.mean()
        gain = float(xc @ (z - z.mean()) / (xc @ xc))
        offset = float(z.mean() - gain * x.mean())
        rows = row_bank == b
        calibrated[rows] = (y[rows] - offset) / gain
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


def _forward(study):
    src = np.asarray(study['source_modes'], dtype=float)
    m, nx, ny = src.shape
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx = v.mean(axis=1)
    vy = v.mean(axis=0) - v.mean()
    interaction = np.linalg.norm(v - vx[:, None] - vy[None, :])
    norm = np.linalg.norm(v)
    eddy = (interaction > 1e-4 * norm or
            np.any(np.asarray(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']))
    factor = 1.45 if eddy else 1.0
    ax = _build_1d(study['Lx'], nx, factor * np.asarray(study['kappa_x']),
                   study['vel_x'], study['robin_x'], study['reaction_x'])
    by = _build_1d(study['Ly'], ny, factor * np.asarray(study['kappa_y']),
                   study['vel_y'], study['robin_y'], study['reaction_y'])
    shifts = study['mu_shifts']
    si, sj = np.asarray(study['sensor_ij']).T
    blocks = []
    if eddy or interaction > 1e-6 * norm:
        # Keep the exact reaction in the threshold interval without forcing eddy diffusion.
        base = (kron(csc_matrix(ax), eye(ny), format='csc') +
                kron(eye(nx), csc_matrix(by), format='csc') + diags(v.ravel(), format='csc'))
        ident = eye(nx * ny, format='csc')
        rhs = src.reshape(m, -1).T
        for mu in shifts:
            fields = splu(base + mu * ident).solve(rhs).T.reshape(src.shape)
            blocks.append(fields[:, si, sj].T)
    else:
        a, qx = schur(ax + np.diag(vx), output='real')
        b, qy = schur(by + np.diag(vy), output='real')
        rhs = (qx.T @ src) @ qy
        for mu in shifts:
            shifted = a + mu * np.eye(nx)
            fields = np.empty_like(src)
            for j, load in enumerate(rhs):
                u, scale, info = dtrsyl(shifted, b, load, tranb='T')
                if info:
                    raise np.linalg.LinAlgError('Ill-conditioned shifted transport operator')
                fields[j] = (qx @ (u / scale)) @ qy.T
            blocks.append(fields[:, si, sj].T)
    return np.concatenate(blocks, axis=0), fields


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
