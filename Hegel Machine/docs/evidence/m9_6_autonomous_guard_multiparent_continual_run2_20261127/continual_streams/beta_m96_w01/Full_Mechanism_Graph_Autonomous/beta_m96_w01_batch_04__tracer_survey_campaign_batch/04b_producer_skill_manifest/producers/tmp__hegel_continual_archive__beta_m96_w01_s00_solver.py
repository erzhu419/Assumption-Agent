"""Laminar tracer tomography with affine calibration and a box-simplex certificate."""

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl


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
            raise ValueError(f'Bank {b} needs calibration at two distinct reference levels')
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
    v = np.asarray(study['V_reaction_2d'], dtype=float)
    vx, vy = v[:, 0], v[0, :] - v[0, 0]
    if (not np.allclose(v, vx[:, None] + vy[None, :], rtol=1e-10, atol=1e-12)
            or max(study['past_vortex_circulation_trace']) > study['critical_vortex_circulation']):
        raise ValueError('The laminar solver requires additive reaction and subcritical circulation')
    axes = tuple(_build_1d(study['L' + a], study['n' + a], study['kappa_' + a],
                           study['vel_' + a], study['robin_' + a], study['reaction_' + a] + r)
                 for a, r in zip('xy', (vx, vy)))
    ax, qx = schur(axes[0], output='real')
    by, qy = schur(axes[1], output='real')
    src = np.asarray(study['source_modes'], dtype=float)
    rhs = (qx.T @ src) @ qy
    si, sj = np.asarray(study['sensor_ij']).T
    diag = np.diag_indices_from(ax)
    blocks = []
    for mu in study['mu_shifts']:
        shifted = ax.copy()
        shifted[diag] += mu
        fields = np.empty_like(src)
        for m, f in enumerate(rhs):
            x, scale, info = dtrsyl(shifted, by, f, tranb='T')
            if info:
                raise np.linalg.LinAlgError('Ill-conditioned shifted transport operator')
            fields[m] = (qx @ (x / scale)) @ qy.T
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


def solve_beta_m96_w01_s00(study: dict) -> dict:
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
