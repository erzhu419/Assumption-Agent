import numpy as np


def _design(x, *features):
    return np.column_stack((np.ones(len(x)), *features))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel):
    design = _design(panel['u'], panel['u'])
    target = panel['v']
    coeff = _fit(design, target)
    for _ in range(12):
        residual = np.linalg.norm(design @ coeff - target, axis=1)
        scale = max(np.median(residual), np.finfo(float).eps)
        weight = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(residual, scale)))
        coeff = _fit(design * weight[:, None], target * weight[:, None])
    residual = np.linalg.norm(design @ coeff - target, axis=1)
    inlier = residual <= 5 * max(np.median(residual), np.finfo(float).eps)
    return _fit(design[inlier], target[inlier])


def _v1_basis(x):
    return _design(x, x, np.tanh(2.274 * x))


def _v2_basis(x):
    return _design(x, np.cos(2.4 * x))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    x = np.concatenate((train['x'], test['x']), axis=0)
    calibration = _calibrate(study['panel_a'])
    u = np.linalg.solve(calibration[1:].T, (x - calibration[0]).T).T
    b, c = study['panel_b'], study['panel_c']
    b_fit = _fit(_design(b['u'], b['u']), b['v'])
    v1_fit = _fit(_v1_basis(c['u']), c['v1'])
    v2_fit = _fit(_v2_basis(c['u']), c['v2'])
    h = np.column_stack((_design(u, u) @ b_fit,
                         _v1_basis(u) @ v1_fit,
                         _v2_basis(u) @ v2_fit))
    return {'h': h, 'n_train': np.array(len(train['x'])),
            'y': train['y'], 'traj': train['traj']}
