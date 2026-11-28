"""Recover latent coordinates from the paired calibration panels."""
import numpy as np


def _panel_a(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    pair = (np.arange(u.shape[1]) + 2) % u.shape[1]
    # Remove the shared nonlinear terms before estimating the channel gains.
    w = v - 0.18 * np.tanh(1.8 * u) + 0.32 * np.sin(2.4 * u[:, pair])
    i, j = np.triu_indices(len(u), 1)
    gain = np.empty(u.shape[1])
    offset = np.empty_like(gain)
    for k in range(u.shape[1]):
        du = u[j, k] - u[i, k]
        dw = w[j, k] - w[i, k]
        good = np.abs(du) > 0.05 * np.ptp(u[:, k])
        slope = np.median(dw[good] / du[good])
        intercept = np.median(w[:, k] - slope * u[:, k])
        error = np.abs(w[:, k] - intercept - slope * u[:, k])
        keep = error <= max(0.001, 6 * np.median(error))
        design = np.column_stack((np.ones(keep.sum()), u[keep, k]))
        offset[k], gain[k] = np.linalg.lstsq(design, w[keep, k], rcond=None)[0]

    # Every Jacobian is strictly diagonally dominant for the calibrated gains.
    # Backtracking gives Newton a safe route even around the sinusoid's turns.
    rhs = x - offset
    z = rhs / gain
    idx = np.arange(u.shape[1])

    def residual(q):
        return q * gain + 0.18 * np.tanh(1.8 * q) - 0.32 * np.sin(2.4 * q[:, pair]) - rhs

    for _ in range(40):
        r = residual(z)
        if np.max(np.abs(r), initial=0.0) < 1e-12:
            break
        th = np.tanh(1.8 * z)
        jac = np.zeros((len(z), len(idx), len(idx)))
        jac[:, idx, idx] = gain + 0.324 * (1 - th * th)
        jac[:, idx, pair] -= 0.768 * np.cos(2.4 * z[:, pair])
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        norm = np.sum(r * r, axis=1)
        size = np.ones((len(z), 1))
        trial = z - step
        for _ in range(15):
            rt = residual(trial)
            bad = np.sum(rt * rt, axis=1) > norm
            if not np.any(bad):
                break
            size[bad] *= 0.5
            trial[bad] = z[bad] - size[bad] * step[bad]
        z = trial
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    xt = np.asarray(study['test']['x'], dtype=float)
    latent = _panel_a(study['panel_a'], np.vstack((x, xt)))
    panel = study['panel_b']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    design = np.column_stack((np.ones(len(u)), u))
    coef = np.linalg.lstsq(design, v, rcond=None)[0]
    z = latent @ coef[1:] + coef[0]
    return {'z': z[:len(x)].copy(), 'zt': z[len(x):].copy()}
