import numpy as np
from scipy.optimize import root


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _pair(x):
    return x * np.roll(x, -1, axis=1)


def _smooth(x):
    k = 1.1
    return x + (np.logaddexp(k * x, -k * x) - np.log(2.0)) / k


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    n, d = u.shape
    nxt = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, nxt * (1 + u), u**3), axis=-1)

    def fit(rows, weights):
        return np.stack([
            np.linalg.lstsq(design[rows, i] * weights[:, None],
                            v[rows, i] * weights, rcond=None)[0]
            for i in range(d)
        ])

    rows = np.arange(n)
    weights = np.ones(n)
    for scale in (1.0, 0.1, 0.01, 0.001):
        for _ in range(8):
            coef = fit(rows, weights)
            residual = np.linalg.norm(v - np.einsum('nij,ij->ni', design, coef), axis=1)
            weights = 1 / np.hypot(1, residual / scale)
    clean = np.argsort(residual)[:-3]
    return fit(clean, np.ones(len(clean)))


def _invert(x, coef):
    offset, slope, coupling, cubic = coef.T
    i = np.arange(len(slope))
    j = (i + 1) % len(i)
    linear = np.diag(slope)
    linear[i, j] = coupling

    def forward(z):
        return offset + slope * z + coupling * z[j] * (1 + z) + cubic * z**3

    def jac(z):
        mat = np.diag(slope + coupling * z[j] + 3 * cubic * z**2)
        mat[i, j] = coupling * (1 + z)
        return mat

    initial = np.linalg.solve(linear, (x - offset).T).T
    return np.stack([root(lambda z: forward(z) - row, guess, jac=jac, tol=1e-10).x
                     for row, guess in zip(x, initial)])


def _features(z, b_coef, c_coef, d_coef):
    b = _affine(z) @ b_coef
    c = _affine(_pair(z)) @ c_coef
    d = _affine(_smooth(z)) @ d_coef
    interaction = (c[:, :, None] * d[:, None, :]).reshape(len(z), -1)
    return np.column_stack((b, c, d, interaction))


def run_node(inputs: dict) -> dict:
    study = inputs.get('__input__', inputs)
    train, test = study['train'], study['test']
    a_coef = _calibrate(study['panel_a'])
    panel_b, panel_c = study['panel_b'], study['panel_c']
    b_coef = np.linalg.lstsq(_affine(panel_b['u']), panel_b['v'], rcond=None)[0]
    c_coef = np.linalg.lstsq(_affine(_pair(panel_c['u'])), panel_c['v1'], rcond=None)[0]
    d_coef = np.linalg.lstsq(_affine(_smooth(panel_c['u'])), panel_c['v2'], rcond=None)[0]
    x = _features(_invert(train['x'], a_coef), b_coef, c_coef, d_coef)
    xp = _features(_invert(test['x'], a_coef), b_coef, c_coef, d_coef)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-10)
    x, xp = (x - center) / scale, (xp - center) / scale

    # Fit the two leading time profiles jointly with the scalar response.
    traj = train['traj']
    _, _, vt = np.linalg.svd(traj, full_matrices=False)
    profiles = vt[:2].T
    target = np.column_stack((train['y'], traj @ profiles))
    mean = target.mean(axis=0)
    coef = np.linalg.solve(x.T @ x + 1e-5 * len(x) * np.eye(x.shape[1]),
                           x.T @ (target - mean))
    pred = xp @ coef + mean
    y_hat = pred[:, 0]
    state_pred = pred[:, 1:] @ profiles.T
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
