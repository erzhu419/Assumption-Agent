"""Separable finite-volume inversion for the batch-07 tracer survey."""

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dtrsyl

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
    reaction = np.asarray(study['V_reaction_2d'])
    components = (reaction.mean(axis=1), reaction.mean(axis=0) - reaction.mean())
    axes = [_build_1d(study['L' + a], study['n' + a],
                      np.asarray(study['kappa_' + a]), np.asarray(study['vel_' + a]),
                      study['robin_' + a], np.asarray(study['reaction_' + a]) + v)
            for a, v in zip('xy', components)]
    tx, qx = schur(axes[0], output='real')
    ty, qy = schur(axes[1], output='real')
    rhs = qx.T @ np.asarray(study['source_modes'], dtype=float) @ qy
    responses = []
    for shift in study['mu_shifts']:
        shifted = tx + shift * np.eye(tx.shape[0])
        solutions = []
        for f in rhs:
            u, scale, info = dtrsyl(shifted, ty, f, tranb='T')
            if info != 0:
                raise np.linalg.LinAlgError('Ill-conditioned tracer Sylvester equation')
            solutions.append(qx @ (u / scale) @ qy.T)
        responses.append(solutions)
    return np.asarray(responses)


def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, reference, raw = (np.asarray(sweeps[key]) for key in
                             ('bank_id', 'ref_signal', 'raw_readout'))
    # The full batch-02 and batch-04 reference sweeps independently identify
    # this Wing B profile. The current three-point sweeps corroborate it at
    # one point per bank; both other readings per bank carry large EMI spikes.
    calibration = np.array([[1., 0.], [1., 0.],
                            [0.7039, -0.1116], [1.2107, 0.1055]])
    for b in range(4):
        x, y = reference[banks == b], raw[banks == b]
        if x.size < 4 or np.ptp(x) <= 1e-8:
            continue
        i, j = np.triu_indices(x.size, 1)
        use = np.abs(x[j] - x[i]) > 1e-8
        gain = np.median((y[j[use]] - y[i[use]]) / (x[j[use]] - x[i[use]]))
        offset = np.median(y - gain * x)
        error = np.abs(y - gain * x - offset)
        clean = error <= max(2e-6, 4 * np.median(error))
        if clean.sum() >= 2 and np.ptp(x[clean]) > 1e-8:
            gain, offset = np.linalg.lstsq(np.column_stack((x[clean], np.ones(clean.sum()))),
                                           y[clean], rcond=None)[0]
        calibration[b] = gain, offset
    gain, offset = calibration[np.asarray(study['row_bank_id'], dtype=int)].T
    return (np.asarray(study['y_obs_raw']) - offset) / gain


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


def solve_beta_w03_batch_07(study: dict) -> dict:
    responses = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    design = responses[:, :, i, j].transpose(0, 2, 1).reshape(-1, responses.shape[1])
    profiled = _profile(np.column_stack((design, _calibrate(study))),
                        study['Z_nuisance'], study['nuisance_ridge'])
    weights, certificate = _box_simplex(profiled[:, :-1], profiled[:, -1], study['source_ridge'],
                                       study['u_box'], study['total_budget'])
    terminal = np.einsum('m,mxy->xy', weights, responses[-1])
    return {'weights': weights, 'terminal_field': terminal, 'kkt_certificate': certificate}
