import numpy as np


def _fit_panel_a(u, v):
    neighbor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1).transpose(1, 0, 2)

    def fit(weights):
        w = np.sqrt(weights)
        return np.stack([np.linalg.lstsq(d * w[:, None], y * w, rcond=None)[0]
                         for d, y in zip(design, v.T)])

    weights = np.ones(len(u))
    for _ in range(12):
        coef = fit(weights)
        residual = np.linalg.norm(v - np.einsum('ijk,ik->ji', design, coef), axis=1)
        scale = max(np.median(residual), 1e-4)
        weights = 1.0 / (1.0 + (residual / scale)**2)
    # The calibration panel contains three grossly corrupted rows.
    weights[:] = 1.0
    weights[np.argsort(residual)[-3:]] = 0.0
    return fit(weights)


def _invert_panel_a(x, coef):
    bias, linear, cross, product, cubic = coef.T
    idx = np.arange(len(bias))
    next_idx = (idx + 1) % len(idx)
    affine = np.diag(linear)
    affine[idx, next_idx] = cross
    z = np.linalg.solve(affine, (x - bias).T).T

    def residual(q):
        neighbor = q[:, next_idx]
        return bias + linear*q + (cross + product*q)*neighbor + cubic*q**3 - x

    for _ in range(30):
        r = residual(z)
        error = np.max(np.abs(r), axis=1)
        if error.max() < 1e-12:
            break
        jac = np.broadcast_to(affine, (len(z), *affine.shape)).copy()
        jac[:, idx, idx] += product*z[:, next_idx] + 3*cubic*z*z
        jac[:, idx, next_idx] += product*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = z - step
        for _ in range(12):
            worse = np.max(np.abs(residual(trial)), axis=1) > error
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        z = trial
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, c = study['panel_a'], study['panel_c']
    coef = _fit_panel_a(a['u'], a['v'])
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _invert_panel_a(x, coef)
    return {'z': z, 'u': c['u'], 'v1': c['v1'], 'v2': c['v2']}
