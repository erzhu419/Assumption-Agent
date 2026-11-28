"""Calibrate the sensor and construct additive kernel features from training observations."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    distance = np.linalg.norm(v - u, axis=1)
    med = np.median(distance)
    mad = np.median(np.abs(distance - med))
    keep = distance <= med + 8 * max(mad, 1e-6)
    u, v = u[keep], v[keep]
    uc = u - u.mean(axis=0)
    scale = (uc * (v - v.mean(axis=0))).sum(axis=0) / (uc * uc).sum(axis=0)
    return v.mean(axis=0) - scale * u.mean(axis=0), scale


def _kernel_features(x, xt, scores):
    """Select coordinate length scales by marginal likelihood on training rows."""
    n, d = x.shape
    delta = (x[:, None, :] - x[None, :, :]) ** 2
    delta_test = (xt[:, None, :] - x[None, :, :]) ** 2
    linear = 1 + x @ x.T
    linear_test = 1 + xt @ x.T
    eye = np.eye(n)

    def objective(h):
        scaled = delta * np.exp(-2 * h[:d])
        parts = np.exp(h[d:2*d] - 0.5 * scaled)
        noise = np.exp(h[-1])
        K = linear + parts.sum(axis=2) + noise * eye
        C = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(C, y, check_finite=False)
        W = cho_solve(C, eye, check_finite=False) - np.outer(alpha, alpha)
        value = 0.5 * (y @ alpha) + np.log(np.diag(C[0])).sum()
        grad = 0.5 * np.r_[np.einsum('ij,ijd,ijd->d', W, parts, scaled),
                            np.einsum('ij,ijd->d', W, parts), noise * np.trace(W)]
        return value, grad

    train_blocks, test_blocks = [], []
    bounds = [(-4., 2.)] * d + [(-10., 4.)] * d + [(-12., 1.)]
    for score in scores.T:
        y = score / max(np.std(score), 1e-8)
        starts = [np.r_[np.full(d, ell), np.zeros(d), -5.] for ell in (-2., -1., 0.)]
        fits = [minimize(objective, h, jac=True, bounds=bounds, method='L-BFGS-B',
                         options={'maxiter': 180}) for h in starts]
        h = min(fits, key=lambda r: r.fun).x
        inv_length2, log_amp = np.exp(-2 * h[:d]), h[d:2*d]
        train_blocks.append(linear + np.exp(log_amp - 0.5 * delta * inv_length2).sum(axis=2)
                            + np.exp(h[-1]) * eye)
        test_blocks.append(linear_test + np.exp(log_amp - 0.5 * delta_test * inv_length2).sum(axis=2))
    return train_blocks, test_blocks


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    shift, scale = _calibrate(study['panel_a'])
    x, xt = ((np.asarray(split['x'], dtype=float) - shift) / scale
             for split in (train, study['test']))
    traj = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    scores = np.column_stack((np.asarray(train['y'], dtype=float), traj @ modes.T))
    K, Kt = _kernel_features(x, xt, scores)
    return {'readout_train': K[0], 'readout_test': Kt[0],
            'state_train': np.stack(K[1:]), 'zt': np.concatenate(Kt[1:], axis=1),
            'modes': modes}
