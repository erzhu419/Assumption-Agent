import numpy as np


def _fit_panel_a(panel):
    u, v = panel['u'], panel['v']
    nxt = np.roll(u, -1, axis=1)
    # Each output has a linear gain, a cubic self term, and coupling to its successor.
    design = np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1).transpose(1, 0, 2)
    targets = v.T[..., None]

    def fit(weights):
        return np.stack([np.linalg.lstsq(d * weights[:, None], y * weights[:, None], rcond=None)[0][:, 0]
                         for d, y in zip(design, targets)])

    weights = np.ones(len(u))
    for _ in range(25):
        coeff = fit(weights)
        residual = np.linalg.norm((design @ coeff[..., None])[..., 0].T - v, axis=1)
        scale = max(1.5 * np.median(residual), 1e-10)
        weights = 1 / np.hypot(1, residual / scale)
    # Refit without the isolated gross errors so their magnitude cannot bias the gains.
    return fit((residual <= 8 * scale).astype(float))


def _invert_panel_a(x, coeff):
    bias, gain, cross, pair, cubic = coeff.T
    j = np.arange(x.shape[1])
    nxt = (j + 1) % len(j)
    linear = np.diag(gain)
    linear[j, nxt] = cross
    z = np.linalg.solve(linear, (x - bias).T).T

    def residual(z):
        return bias + gain * z + (cross + pair * z) * z[:, nxt] + cubic * z**3 - x

    for _ in range(30):
        r = residual(z)
        if np.max(np.abs(r)) < 1e-12:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = gain + pair * z[:, nxt] + 3 * cubic * z**2
        jac[:, j, nxt] = cross + pair * z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        new = z - step
        old_norm = np.sum(r**2, axis=1)
        for _ in range(12):
            worse = np.sum(residual(new)**2, axis=1) > old_norm
            if not worse.any():
                break
            step[worse] *= 0.5
            new[worse] = z[worse] - step[worse]
        z = new
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    coeff = _fit_panel_a(study['panel_a'])
    u, v = study['panel_b']['u'], study['panel_b']['v']
    design = np.column_stack((np.ones(len(u)), u))
    affine = np.linalg.lstsq(design, v, rcond=None)[0]

    def calibrate(x):
        return _invert_panel_a(x, coeff) @ affine[1:] + affine[0]

    return {'z_tr': calibrate(train['x']), 'z_te': calibrate(test['x']),
            'y': train['y'], 'traj': train['traj'], 'eval_times': study['eval_times']}
