import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _parameters(panel):
    u, v = panel['u'], panel['v']
    successor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, successor, u * successor, u**3), axis=2).transpose(1, 0, 2)
    weights = np.ones(len(u))
    for _ in range(30):
        coef = np.array([np.linalg.lstsq(d * weights[:, None], col * weights, rcond=None)[0] for d, col in zip(design, v.T)])
        residual = np.linalg.norm(v.T - np.einsum('cnp,cp->cn', design, coef), axis=0)
        scale = max(1e-12, 2 * np.median(residual))
        weights = np.sqrt(np.minimum(1, scale / np.maximum(residual, 1e-12)))
    good = np.argsort(residual)[:len(u)-3]
    return np.array([np.linalg.lstsq(d[good], col[good], rcond=None)[0] for d, col in zip(design, v.T)]).T


def _forward(u, coef):
    offset, linear, cross, pair, cubic = coef
    successor = np.roll(u, -1, axis=1)
    return offset + linear*u + cross*successor + pair*u*successor + cubic*u**3


def _invert(v, coef):
    offset, linear, cross, pair, cubic = coef
    i = np.arange(v.shape[1]); j = (i + 1) % len(i)
    affine = np.diag(linear); affine[i, j] = cross
    u = np.linalg.solve(affine, (v-offset).T).T
    jac = np.zeros((len(v), len(i), len(i)))
    for _ in range(30):
        r = _forward(u, coef) - v
        if np.max(np.abs(r)) < 1e-11:
            break
        jac[:, i, i] = linear + pair*u[:, j] + 3*cubic*u*u
        jac[:, i, j] = cross + pair*u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        alpha = np.ones((len(u), 1))
        trial = u - step
        for _ in range(16):
            worse = np.linalg.norm(_forward(trial, coef) - v, axis=1) > np.linalg.norm(r, axis=1)
            if not worse.any():
                break
            alpha[worse] *= .5
            trial = u - alpha*step
        u = trial
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    panel_a, panel_b = study['panel_a'], study['panel_b']
    a_coef = _parameters(panel_a)
    b_coef = np.linalg.lstsq(_affine(panel_b['u']), panel_b['v'], rcond=None)[0]
    # Undo the Panel A measurement map, then express the recovered points in Panel B coordinates.
    z = _affine(_invert(train['x'], a_coef)) @ b_coef
    z_test = _affine(_invert(test['x'], a_coef)) @ b_coef
    return {'z': z, 'z_test': z_test, 'y': train['y'], 'traj': train['traj']}
