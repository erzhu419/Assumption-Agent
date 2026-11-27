"""Calibrated laminar tracer inversion and box-simplex certificates."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _bank_sweep(sweep, bank):
    mask = np.asarray(sweep['bank_id']) == bank
    return (np.asarray(sweep['ref_signal'], dtype=float)[mask],
            np.asarray(sweep['raw_readout'], dtype=float)[mask])


def _line(x, y):
    dx = x - x.mean()
    gain = dx @ (y - y.mean()) / (dx @ dx)
    return gain, y.mean() - gain * x.mean()


def _historical_sweep(bank, history):
    for sweep in reversed(history):
        x, y = _bank_sweep(sweep, bank)
        if x.size >= 2 and np.ptp(x) > 0:
            return x, y
    raise ValueError(f'No calibration at distinct reference levels for bank {bank}')


def _historical_line(bank, history):
    return _line(*_historical_sweep(bank, history))


def _sparse_line(x, y, bank, history):
    # Sparse readings cannot independently distinguish gain drift from EMI.
    # Use the archived response span to screen gross deviations, and update
    # the offset with current readings that agree with the archived line.
    hx, hy = _historical_sweep(bank, history)
    gain, offset = _line(hx, hy)
    if not x.size:
        return gain, offset
    span = abs(gain) * np.ptp(hx)
    delta = y - (gain * x + offset)
    keep = np.abs(delta) <= 0.05 * span
    if x.size >= 2 and np.ptp(x) > 0:
        current_gain, current_offset = _line(x, y)
        # Retain a coherent sparse update even if its endpoint readings have
        # drifted away from the archived line. A lone distant reading does not
        # establish a new gain or offset.
        if (current_gain * gain > 0 and
                abs(current_gain - gain) <= 0.20 * abs(gain) and
                abs(current_offset - offset) <= 0.06 * span and
                np.max(np.abs(y - (current_gain * x + current_offset))) <= 0.02 * span):
            return current_gain, current_offset
    if np.count_nonzero(keep) >= 2 and np.ptp(x[keep]) > 0:
        return _line(x[keep], y[keep])
    if keep.any():
        offset += np.median(delta[keep])
    return gain, offset


def _calibrate(study):
    banks = np.asarray(study['row_bank_id'])
    raw = np.asarray(study['y_obs_raw'], dtype=float)
    obs = np.empty_like(raw)
    history = None
    for bank in np.unique(banks):
        x, y = _bank_sweep(study['sensor_calibration_sweeps'], bank)
        if x.size < 4 or np.ptp(x) == 0:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            gain, offset = _sparse_line(x, y, bank, history)
        else:
            keep = np.ones(x.size, dtype=bool)
            if x.size >= 4:
                # Score pairwise lines by consensus before refitting the inliers.
                i, j = np.triu_indices(x.size, 1)
                distinct = x[i] != x[j]
                i, j = i[distinct], j[distinct]
                gains = (y[j] - y[i]) / (x[j] - x[i])
                offsets = y[i] - gains * x[i]
                residuals = np.abs(y - gains[:, None] * x - offsets[:, None])
                # Strict majority: a line through only the defining pair is insufficient.
                rank = x.size // 2
                scores = np.partition(residuals, rank, axis=1)[:, rank]
                best = scores.argmin()
                cutoff = max(8 * scores[best], 32 * np.finfo(float).eps * max(1., np.max(np.abs(y))))
                inliers = residuals[best] <= cutoff
                if np.ptp(x[inliers]) > 0:
                    keep = inliers
            gain, offset = _line(x[keep], y[keep])
        obs[banks == bank] = (raw[banks == bank] - offset) / gain
    return obs


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


def solve_beta_m96_w01_batch_01(study: dict) -> dict:
    y = _calibrate(study)
    design, last_fields = _forward(study)
    profiled = _residualize(study['Z_nuisance'], study['nuisance_ridge'],
                            np.column_stack((design, y)))
    s, y = profiled[:, :-1], profiled[:, -1]
    ridge, cap = (float(study[k]) for k in ('source_ridge', 'u_box'))
    weights, certs = [], []
    for budget in map(float, study['budget_grid']):
        w, nu = _fit_weights(s, y, ridge, cap, budget)
        w = np.clip(w, 0., cap)
        res = s @ w - y
        g = s.T @ res + ridge * w
        primal = float(0.5 * (res @ res + ridge * (w @ w)))
        reduced = g - nu
        dual = primal - g @ w + nu * budget + cap * np.minimum(reduced, 0.).sum()
        weights.append(w)
        certs.append(dict(weights=w.tolist(), nu_sum=nu,
                          lam_lower=np.maximum(reduced, 0.).tolist(),
                          lam_upper=np.maximum(-reduced, 0.).tolist(),
                          primal_objective=primal, dual_lower_bound=min(primal, float(dual))))
    return dict(weights_by_budget=np.asarray(weights),
                terminal_field_mid_budget=np.einsum('m,mij->ij', weights[1], last_fields),
                kkt_certificates=certs)
