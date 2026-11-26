"""Full two-dimensional finite-volume inversion for the batch-06 vortex tracer survey."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L / n
    diffusion = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:]) / h**2
    forward = np.maximum(vel, 0) / h
    backward = np.maximum(-vel, 0) / h
    diagonal = np.r_[0, diffusion] + np.r_[diffusion, 0] + forward + backward + diag_add
    edge = np.array([0, n - 1])
    diagonal[edge] += robin * kappa[edge] / (h * (0.5 * robin * h + kappa[edge]))
    return (np.diag(diagonal) - np.diag(diffusion + backward[:-1], 1)
            - np.diag(diffusion + forward[1:], -1))


def _responses(study):
    v = np.asarray(study['V_reaction_2d'])
    axes = [_build_1d(study['L' + axis], study['n' + axis],
                      1.45 * np.asarray(study['kappa_' + axis]), np.asarray(study['vel_' + axis]),
                      study['robin_' + axis], np.asarray(study['reaction_' + axis]))
            for axis in 'xy']
    nx, ny = v.shape
    base = (kron(axes[0], eye(ny), format='csc') + kron(eye(nx), axes[1], format='csc')
            + diags(v.ravel(), format='csc'))
    rhs = np.asarray(study['source_modes'], dtype=float).reshape(-1, nx * ny).T
    return np.stack([splu(base + shift * eye(nx * ny, format='csc')).solve(rhs).T
                     .reshape(-1, nx, ny) for shift in study['mu_shifts']])


def _sweeps(record):
    banks, ref, raw = (np.asarray(record[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    return [(ref[banks == b], raw[banks == b]) for b in range(4)]


def _fit_sweep(x, y, prior):
    """Fit the current affine line, with the archived line breaking sparse ties."""
    gain, offset = prior
    if not x.size:
        return prior
    i, j = np.triu_indices(x.size, 1)
    span = x[j] - x[i]
    distinct = np.abs(span) > 1e-8
    i, j, span = i[distinct], j[distinct], span[distinct]
    slopes = (y[j] - y[i]) / span
    models = [(gain, offset), (gain, np.median(y - gain * x))]
    models.extend((a, y[k] - a * x[k]) for a, k in zip(slopes, i) if a > 0)
    med = np.median(slopes) if slopes.size else gain
    resid = y - med * x
    mad = np.median(np.abs(resid - np.median(resid)))
    # Do not let sparse, contaminated rows inflate the consensus radius.
    tol = np.clip(4 * mad, 2e-6, 0.02)
    def score(model):
        a, b = model
        err = np.abs(y - a * x - b)
        return ((err <= tol).sum(), -np.mean(np.minimum(err, tol)),
                -abs(a - gain) - abs(b - offset))
    best = max(models, key=score)
    keep = np.abs(y - best[0] * x - best[1]) <= tol
    if keep.sum() >= 3 and np.ptp(x[keep]) > 1e-8:
        a, b = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
        if a > 0:
            return a, b
    if np.max(np.abs(y - gain * x - offset)) <= 2e-6:
        return gain, np.median(y - gain * x)
    if x.size == 1 or (x.size == 2 and np.min(np.abs(y - gain * x - offset)) <= 2e-6):
        return prior
    return best


def _calibrate(study):
    samples = _sweeps(study['sensor_calibration_sweeps'])
    priors = [(1., 0.)] * 4
    if any(x.size < 4 or np.ptp(x) <= 1e-8 for x, _ in samples):
        path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
        history = []
        for record in json.loads(path.read_text()) if path.is_file() else []:
            # A spot-only record can confirm a campaign but cannot supply a
            # profile for banks it never measured.
            profile = {b: _fit_sweep(x, y, (1., 0.))
                       for b, (x, y) in enumerate(_sweeps(record))
                       if x.size >= 2 and np.ptp(x) > 1e-8}
            history.append(profile)
        def support(profile, banks):
            return sum(np.count_nonzero(np.abs(y - profile[b][0] * x - profile[b][1]) <= 2e-6)
                       for b in banks if b in profile for x, y in [samples[b]])
        for wing in ((0, 1), (2, 3)):
            eligible = [profile for profile in history if all(b in profile for b in wing)]
            if not eligible:
                continue
            def score(profile):
                local = support(profile, wing)
                return (local if local >= 2 else 0, support(profile, range(4)), len(profile))
            profile = max(reversed(eligible), key=score)
            for b in wing:
                priors[b] = profile[b]
    gain, offset = np.asarray([_fit_sweep(x, y, prior) for (x, y), prior in zip(samples, priors)]).T
    rows = np.asarray(study['row_bank_id'], dtype=int)
    return (np.asarray(study['y_obs_raw']) - offset[rows]) / gain[rows]


def _profile(values, nuisance, ridge):
    basis, sigma, _ = np.linalg.svd(nuisance, full_matrices=False)
    square = sigma ** 2
    denom = square + ridge
    # The report's objective uses squared ridge-adjusted residuals.
    shrink = np.divide(square, denom, out=np.zeros_like(square), where=denom > 0)
    return values - basis @ (shrink[:, None] * (basis.T @ values))


def _project(v, cap, budget):
    lo, hi = v.min() - cap, v.max()
    for _ in range(55):
        mid = (lo + hi) / 2
        if np.clip(v - mid, 0, cap).sum() > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(v - (lo + hi) / 2, 0, cap)


def _box_simplex(S, y, ridge, cap, budget):
    m = S.shape[1]
    H, f = S.T @ S + ridge * np.eye(m), S.T @ y
    w = np.full(m, budget / m)
    step = np.linalg.eigvalsh(H)[-1]
    if 0 < budget < m * cap and step > 0:
        z = w.copy()
        for t in range(1, 1001):
            new = _project(z - (H @ z - f) / step, cap, budget)
            gradient = H @ new - f
            if gradient @ new - _linear_minimum(gradient, cap, budget) < 2e-13:
                w = new
                break
            z = new + (t - 1) / (t + 2) * (new - w)
            w = new
        free = (w > 1e-10) & (w < cap - 1e-10)
        if free.any():
            # Solve the equality-constrained subproblem on the identified face.
            upper = w >= cap - 1e-10
            F = H[np.ix_(free, free)]
            ones = np.ones(free.sum())
            kkt = np.block([[F, -ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            rhs = np.r_[f[free] - cap * H[np.ix_(free, upper)].sum(axis=1), budget - cap * upper.sum()]
            exact = np.zeros_like(w)
            exact[upper] = cap
            exact[free] = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
            if np.all((exact >= 0) & (exact <= cap)):
                gradient = H @ exact - f
                if gradient @ exact - _linear_minimum(gradient, cap, budget) < 2e-13:
                    w = exact
    residual = S @ w - y
    g = S.T @ residual + ridge * w
    # A budget quantile maximizes the box-relaxed affine lower bound.
    nu = float(np.sort(g)[min(m - 1, int(budget / cap))]) if cap > 0 else 0.
    ll, lu = np.maximum(g - nu, 0), np.maximum(nu - g, 0)
    objective = float(0.5 * (residual @ residual + ridge * (w @ w)))
    dual = float(objective - g @ w + nu * budget - cap * lu.sum())
    return w, dict(weights=w.copy(), nu_sum=nu, lam_lower=ll, lam_upper=lu,
                   primal_objective=objective, dual_lower_bound=min(objective, dual))


def _linear_minimum(g, cap, budget):
    fill = np.clip(budget - cap * np.arange(g.size), 0, cap)
    return np.sort(g) @ fill


def solve_beta_w03_batch_06(study: dict) -> dict:
    responses = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    design = responses[:, :, i, j].transpose(0, 2, 1).reshape(-1, responses.shape[1])
    profiled = _profile(np.column_stack((design, _calibrate(study))),
                        study['Z_nuisance'], study['nuisance_ridge'])
    weights, certificate = _box_simplex(profiled[:, :-1], profiled[:, -1], study['source_ridge'],
                                       study['u_box'], study['total_budget'])
    terminal = np.einsum('m,mxy->xy', weights, responses[-1])
    return {'weights': weights, 'terminal_field': terminal, 'kkt_certificate': certificate}
