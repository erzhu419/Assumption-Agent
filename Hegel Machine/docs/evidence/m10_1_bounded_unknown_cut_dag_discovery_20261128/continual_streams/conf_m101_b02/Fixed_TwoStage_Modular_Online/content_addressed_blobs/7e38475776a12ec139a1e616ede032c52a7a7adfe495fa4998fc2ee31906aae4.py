import numpy as np
from itertools import combinations_with_replacement


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibration(panel):
    # Fit the forward measurement map before using it to correct coordinates.
    design = _affine(panel['u'])
    values = panel['v']
    coef = _fit(design, values)
    for _ in range(20):
        residual = np.linalg.norm(values - design @ coef, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / (2 * scale))
        coef = _fit(design * weight[:, None], values * weight[:, None])
    return coef


def _cross_terms(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _diagonal_quadratic(x):
    return np.column_stack((_affine(x), x*x))


def _cubic(x):
    columns = [_affine(x)]
    for degree in (2, 3):
        columns.extend(np.prod(x[:, indices], axis=1)
                       for indices in combinations_with_replacement(range(x.shape[1]), degree))
    return np.column_stack(columns)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    measured = np.vstack((train['x'], test['x']))
    calibration = _calibration(study['panel_a'])
    x = np.linalg.solve(calibration[1:].T, (measured - calibration[0]).T).T

    b, c = study['panel_b'], study['panel_c']
    fd = _affine(x) @ _fit(_affine(b['u']), b['v'])
    v1 = _cross_terms(x) @ _fit(_cross_terms(c['u']), c['v1'])
    v2 = _diagonal_quadratic(x) @ _fit(_diagonal_quadratic(c['u']), c['v2'])
    return {'g': _cubic(x), 'fd': fd, 'cd': np.column_stack((v1, v2)),
            'y': train['y'], 'traj': train['traj']}
