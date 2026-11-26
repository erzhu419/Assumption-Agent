"""Coupled finite-volume inversion for the batch-08 tracer survey."""
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
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
    nx, ny = study['nx'], study['ny']
    axes = [csc_matrix(_build_1d(study['L'+a], study['n'+a],
                               1.45*np.asarray(study['kappa_'+a]),
                               np.asarray(study['vel_'+a]), study['robin_'+a],
                               np.asarray(study['reaction_'+a]))) for a in 'xy']
    base = (kron(axes[0], eye(ny), format='csc') +
            kron(eye(nx), axes[1], format='csc'))
    reaction = np.asarray(study['V_reaction_2d']).ravel()
    rhs = np.asarray(study['source_modes']).reshape(-1, nx*ny)
    return np.asarray([splu(base + diags(reaction+mu, format='csc')).solve(rhs.T).T
                       .reshape(-1, nx, ny) for mu in study['mu_shifts']])

def _calibrate(study):
    sweeps = study['sensor_calibration_sweeps']
    banks, reference, raw = (np.asarray(sweeps[key]) for key in
                             ('bank_id', 'ref_signal', 'raw_readout'))
    # The current bank-0 checks match the batch-07 Wing A calibration.
    # Wing B retains the profile independently measured in batches 02 and 04.
    calibration = np.array([[1.2602, 0.1597], [0.7999, -0.1105],
                            [0.7039, -0.1116], [1.2107, 0.1055]])
    for b in range(len(calibration)):
        x, y = reference[banks == b], raw[banks == b]
        if x.size < 2:
            continue
        i, j = np.triu_indices(x.size, 1)
        separated = np.abs(x[j] - x[i]) > 1e-8
        i, j = i[separated], j[separated]
        if not i.size:
            continue
        slopes = (y[j] - y[i]) / (x[j] - x[i])
        intercepts = y[i] - slopes * x[i]
        candidates = np.vstack((calibration[b], np.column_stack((slopes, intercepts))))
        errors = np.abs(candidates[:, :1] * x + candidates[:, 1:] - y)
        inliers = (errors <= 3e-6) & (candidates[:, :1] > 0)
        # Sparse spot checks that agree with the archive confirm the profile;
        # refitting them would turn readout noise into gain and offset drift.
        if x.size == 2 and inliers[0].all():
            continue
        clean = inliers[np.argmax(inliers.sum(axis=1))]
        # With three or more checks, a pair alone cannot displace an
        # archive profile corroborated by a different check (batch-07 EMI).
        if x.size >= 3 and clean.sum() == 2 and inliers[0].any():
            continue
        if clean.sum() >= 2 and np.ptp(x[clean]) > 1e-8:
            calibration[b] = np.linalg.lstsq(
                np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None)[0]
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


def solve_beta_w03_batch_08(study: dict) -> dict:
    responses = _responses(study)
    i, j = np.asarray(study['sensor_ij'], dtype=int).T
    design = responses[:, :, i, j].transpose(0, 2, 1).reshape(-1, responses.shape[1])
    profiled = _profile(np.column_stack((design, _calibrate(study))),
                        study['Z_nuisance'], study['nuisance_ridge'])
    weights, certificate = _box_simplex(profiled[:, :-1], profiled[:, -1], study['source_ridge'],
                                       study['u_box'], study['total_budget'])
    terminal = np.einsum('m,mxy->xy', weights, responses[-1])
    return {'weights': weights, 'terminal_field': terminal, 'kkt_certificate': certificate}
