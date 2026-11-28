import numpy as np


def _fit_panel(panel):
    """Fit the forward cyclic calibration on the clean reference coordinates."""
    u, v = panel['u'], panel['v']
    neighbor = np.roll(u, -1, axis=1)
    n, d = u.shape
    eye = np.eye(d)
    offsets = np.broadcast_to(eye, (n, d, d))
    design = np.concatenate((offsets, u[..., None] * eye,
                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)

    def fit(weights):
        return np.linalg.lstsq((design * weights[:, None, None]).reshape(-1, 2*d+3),
                               (v * weights[:, None]).ravel(), rcond=None)[0]

    weights = np.ones(n)
    for _ in range(24):
        residual = np.linalg.norm(design @ fit(weights) - v, axis=1)
        scale = max(0.7 * np.median(residual), 1e-12)
        weights = 1 / np.hypot(1, residual / scale)

    # Refine without shrinkage after separating the three corrupted panel rows.
    clean = np.ones(n)
    clean[np.argsort(residual)[-3:]] = 0
    return fit(clean)


def _invert(x, coefficients):
    d = x.shape[1]
    bias, gain = coefficients[:d], coefficients[d:2*d]
    cross, interaction, cubic = coefficients[2*d:]
    i = np.arange(d)
    j = (i + 1) % d
    linear = np.diag(gain)
    linear[i, j] = cross
    z = np.linalg.solve(linear, (x - bias).T).T
    jac = np.zeros((len(x), d, d))
    for _ in range(25):
        neighbor = z[:, j]
        residual = bias + gain*z + cross*neighbor + interaction*z*neighbor + cubic*z**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, i, i] = gain + interaction*neighbor + 3*cubic*z*z
        jac[:, i, j] = cross + interaction*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= step
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    coefficients = _fit_panel(study['panel_a'])
    calibrated = _invert(np.concatenate((train['x'], test['x']), axis=0), coefficients)
    n = len(train['x'])
    return dict(z_tr=calibrated[:n], z_te=calibrated[n:], panel_b=study['panel_b'],
                eval_times=study['eval_times'], y=train['y'], traj=train['traj'])
