import numpy as np
from scipy.optimize import least_squares


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel_a_inverse(panel, x):
    design = _affine(panel['u'])
    observed = panel['v']
    coef = _fit(design, observed)
    for _ in range(12):
        residual = np.linalg.norm(observed - design @ coef, axis=1)
        scale = max(np.median(residual), 1e-5)
        root_weight = 1 / np.hypot(1, residual / scale)
        coef = _fit(design * root_weight[:, None], observed * root_weight[:, None])
    residual = np.linalg.norm(observed - design @ coef, axis=1)
    keep = np.argsort(residual)[:-3]
    offset, *linear = _fit(design[keep], observed[keep])
    return np.linalg.solve(np.asarray(linear).T, (x - offset).T).T


def _v1_basis(u):
    a, b, c, d = u.T
    return np.column_stack((np.ones(len(u)), a * b, a * d, b * c, c * d))


def _v2_basis(u, frequency):
    return np.column_stack((_affine(u), np.cos(frequency * u)))


def _panel_c_v2(u, observed, x):
    def residual(frequency):
        design = _v2_basis(u, frequency)
        return (design @ _fit(design, observed) - observed).ravel()

    frequency = least_squares(residual, [1.3], bounds=(0.1, 7)).x
    return _v2_basis(x, frequency) @ _fit(_v2_basis(u, frequency), observed)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    x = _panel_a_inverse(study['panel_a'], np.vstack((train['x'], test['x'])))
    b, c = study['panel_b'], study['panel_c']
    vb = _affine(x) @ _fit(_affine(b['u']), b['v'])
    v1 = _v1_basis(x) @ _fit(_v1_basis(c['u']), c['v1'])
    v2 = _panel_c_v2(c['u'], c['v2'], x)
    return {
        'n': np.asarray(len(train['x'])),
        'linear': np.column_stack((np.ones(len(x)), vb, v1, v2)),
        'mixed': (v1[:, :, None] * v2[:, None, :]).reshape(len(x), -1),
        'eval_times': study['eval_times'],
        'y': train['y'],
        'traj': train['traj'],
    }
