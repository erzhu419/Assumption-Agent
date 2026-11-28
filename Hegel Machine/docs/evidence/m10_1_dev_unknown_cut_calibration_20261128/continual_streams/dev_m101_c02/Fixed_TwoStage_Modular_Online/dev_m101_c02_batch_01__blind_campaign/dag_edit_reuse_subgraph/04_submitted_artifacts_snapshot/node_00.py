import numpy as np
from scipy.optimize import least_squares


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    design = np.column_stack((np.ones(len(u)), u))
    coef = _fit(design, v)
    for _ in range(30):
        residual = np.linalg.norm(design @ coef - v, axis=1)
        weight = 1 / np.hypot(1, residual / max(np.median(residual), 1e-9))
        coef = _fit(design * weight[:, None], v * weight[:, None])
    keep = np.argsort(np.linalg.norm(design @ coef - v, axis=1))[:-3]
    return _fit(design[keep], v[keep])


def _features(x, frequency):
    return np.column_stack((np.ones(len(x)), x, np.sin(x * frequency)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    affine = _calibrate(study['panel_a'])
    u, v = study['panel_b']['u'], study['panel_b']['v']

    def residual(frequency):
        design = _features(u, frequency)
        return (design @ _fit(design, v) - v).ravel()

    frequency = least_squares(residual, np.full(u.shape[1], 2.5), bounds=(0.2, 6.0)).x
    coef = _fit(_features(u, frequency), v)

    def transform(x):
        calibrated = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T
        return _features(calibrated, frequency) @ coef

    return {'z_tr': transform(study['train']['x']), 'z_te': transform(study['test']['x'])}
