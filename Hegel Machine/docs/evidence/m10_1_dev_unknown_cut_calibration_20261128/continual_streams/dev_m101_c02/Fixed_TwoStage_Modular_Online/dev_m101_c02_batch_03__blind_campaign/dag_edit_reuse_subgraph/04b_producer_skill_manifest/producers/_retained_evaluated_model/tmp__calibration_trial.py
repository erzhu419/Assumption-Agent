import numpy as np


def _fit_panel_a(u, v):
    nxt = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, nxt, u*nxt, u**3), axis=1)

    def fit(w):
        return np.stack([np.linalg.lstsq(design[:, :, j]*w[:, None], v[:, j]*w, rcond=None)[0] for j in range(u.shape[1])], axis=1)

    weights = np.ones(len(u))
    for _ in range(25):
        coef = fit(weights)
        residual = np.linalg.norm(np.einsum('nkj,kj->nj', design, coef) - v, axis=1)
        scale = max(0.002, np.median(residual))
        weights = scale / np.hypot(scale, residual)
    keep = np.argsort(residual)[:-3]
    weights = np.zeros(len(u))
    weights[keep] = 1.0
    return fit(weights)


def _forward_a(z, coef):
    offset, linear, neighbor, interaction, cubic = coef
    nxt = np.roll(z, -1, axis=1)
    return offset + linear*z + (neighbor + interaction*z)*nxt + cubic*z**3


def _inverse_a(v, coef):
    offset, linear, neighbor, interaction, cubic = coef
    z = np.zeros_like(v)
    i = np.arange(v.shape[1])
    for _ in range(35):
        residual = _forward_a(z, coef) - v
        if np.max(np.abs(residual)) < 1e-11:
            break
        jac = np.zeros((len(v), len(i), len(i)))
        jac[:, i, i] = linear + interaction*np.roll(z, -1, axis=1) + 3*cubic*z*z
        jac[:, i, (i+1) % len(i)] = neighbor + interaction*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        err = np.linalg.norm(residual, axis=1)
        damping = np.ones(len(z))
        for _ in range(15):
            trial = z - damping[:, None]*step
            bad = np.linalg.norm(_forward_a(trial, coef)-v, axis=1) > err
            if not np.any(bad):
                break
            damping[bad] *= 0.5
        z = trial
    return z
