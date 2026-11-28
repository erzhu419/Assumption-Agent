import numpy as np


def _fit_forward(u, v):
    # A cyclic neighbour modulates each coordinate's gain.
    neighbour = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, u**3, (u - 1) * neighbour), axis=-1)

    def fit(weights):
        w = np.sqrt(weights)
        return np.stack([np.linalg.lstsq(design[j] * w[:, None], v[:, j] * w,
                                        rcond=None)[0] for j in range(u.shape[1])])

    design = design.transpose(1, 0, 2)
    weights = np.ones(len(u))
    for _ in range(30):
        coeff = fit(weights)
        error = np.linalg.norm(v - np.einsum('jnp,jp->nj', design, coeff), axis=1)
        scale = max(1.8 * np.median(error), 1e-8)
        weights = 1 / (1 + (error / scale)**2)
    # The calibration protocol contains three corrupted rows.
    weights = np.ones(len(u))
    weights[np.argsort(error)[-3:]] = 0
    return fit(weights)


def _invert(y, coeff):
    offset, gain, cubic, cross = coeff.T
    dim = len(gain)
    rows = np.arange(dim)
    next_rows = (rows + 1) % dim
    linear = np.diag(gain)
    linear[rows, next_rows] = -cross
    x = np.linalg.solve(linear, (y - offset).T).T

    def forward(z):
        return offset + gain * z + cubic * z**3 + cross * (z - 1) * np.roll(z, -1, axis=1)

    for _ in range(40):
        r = forward(x) - y
        norm = np.linalg.norm(r, axis=1)
        if np.max(norm) < 1e-11:
            break
        jac = np.zeros((len(x), dim, dim))
        jac[:, rows, rows] = gain + 3 * cubic * x**2 + cross * np.roll(x, -1, axis=1)
        jac[:, rows, next_rows] = cross * (x - 1)
        delta = np.linalg.solve(jac, r[..., None])[..., 0]
        step = np.ones(len(x))
        for _ in range(18):
            trial = x - step[:, None] * delta
            improved = np.linalg.norm(forward(trial) - y, axis=1) < norm
            if np.all(improved | (norm < 1e-11)):
                break
            step[~improved] *= 0.5
        x[improved] = trial[improved]
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    coeff = _fit_forward(panel['u'], panel['v'])
    train, test = study['train']['x'], study['test']['x']
    calibrated = _invert(np.concatenate((train, test), axis=0), coeff)
    return {'cal_tr': calibrated[:len(train)], 'cal_te': calibrated[len(train):],
            'panel_b': study['panel_b']}
