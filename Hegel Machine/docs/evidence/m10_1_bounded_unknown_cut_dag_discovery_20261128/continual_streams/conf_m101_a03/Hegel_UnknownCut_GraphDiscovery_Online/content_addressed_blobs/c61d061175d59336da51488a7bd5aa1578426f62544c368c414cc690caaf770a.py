import numpy as np


def _basis(u):
    neighbor = np.roll(u, -1, axis=1)
    return np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)


def _fit_forward(u, v):
    # Use one weight per panel row because the contamination is row-wise.
    design = _basis(u)
    weight = np.ones(len(u))

    def fit(w):
        return np.stack([
            np.linalg.lstsq(design[:, j] * w[:, None], v[:, j] * w, rcond=None)[0]
            for j in range(u.shape[1])
        ])

    for _ in range(16):
        coeff = fit(weight)
        residual = np.linalg.norm(np.einsum('njk,jk->nj', design, coeff) - v, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1.0 / np.hypot(1.0, residual / (2.0 * scale))
    return fit(weight)


def _invert(v, coeff):
    if not len(v):
        return v.copy()
    bias, gain, cross, interaction, cubic = coeff.T
    j = np.arange(len(bias))
    k = (j + 1) % len(j)
    linear = np.diag(gain)
    linear[j, k] = cross
    u = np.linalg.solve(linear, (v - bias).T).T

    def error(x):
        neighbor = x[:, k]
        return bias + gain * x + cross * neighbor + interaction * x * neighbor + cubic * x**3 - v

    residual = error(u)
    jac = np.zeros((len(v), len(j), len(j)))
    for _ in range(30):
        if np.max(np.abs(residual)) < 1e-11:
            break
        jac[:, j, j] = gain + interaction * u[:, k] + 3 * cubic * u**2
        jac[:, j, k] = cross + interaction * u
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        old_norm = np.sum(residual**2, axis=1)
        factor = np.ones((len(v), 1))
        for _ in range(16):
            trial = u - factor * step
            trial_residual = error(trial)
            better = np.sum(trial_residual**2, axis=1) < old_norm
            pending = (~better) & (old_norm > 1e-22)
            if not np.any(pending):
                break
            factor[pending] *= 0.5
        u[better] = trial[better]
        residual[better] = trial_residual[better]
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    coeff = _fit_forward(np.asarray(panel['u'], dtype=float), np.asarray(panel['v'], dtype=float))
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    calibrated = _invert(np.concatenate((train, test)), coeff)
    return {'x_tr_cal': calibrated[:len(train)], 'x_te_cal': calibrated[len(train):]}
