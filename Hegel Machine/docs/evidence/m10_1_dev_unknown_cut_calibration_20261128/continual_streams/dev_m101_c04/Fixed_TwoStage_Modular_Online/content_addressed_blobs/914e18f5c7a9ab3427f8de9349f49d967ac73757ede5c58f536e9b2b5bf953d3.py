import numpy as np
from scipy.optimize import least_squares


def _fit_affine(panel):
    u, v = panel['u'], panel['v']
    design = np.column_stack((np.ones(len(u)), u))
    coef = np.linalg.lstsq(design, v, rcond=None)[0]
    # Fit in measurement space so corrupted observations do not become predictors.
    for _ in range(15):
        error = np.linalg.norm(design @ coef - v, axis=1)
        weight = np.maximum(error, 1e-8)**-0.5
        coef = np.linalg.lstsq(design * weight[:, None], v * weight[:, None], rcond=None)[0]
    inliers = np.argsort(np.linalg.norm(design @ coef - v, axis=1))[:-3]
    return np.linalg.lstsq(design[inliers], v[inliers], rcond=None)[0]


def _sine_design(u, frequency):
    return np.column_stack((np.ones(len(u)), u, np.sin(u * frequency)))


def _fit_sine(panel):
    u, v = panel['u'], panel['v']

    def profile(frequency):
        design = _sine_design(u, frequency)
        coef = np.linalg.lstsq(design, v, rcond=None)[0]
        return (design @ coef - v).ravel(), coef

    result = least_squares(lambda f: profile(f)[0], np.full(u.shape[1], 3.0),
                           bounds=(0.1, 8.0), max_nfev=150, gtol=1e-11)
    return result.x, profile(result.x)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    affine = _fit_affine(study['panel_a'])
    frequency, coef = _fit_sine(study['panel_b'])

    def calibrate(x):
        u = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T
        return _sine_design(u, frequency) @ coef

    train = study['train']
    return {'z_tr': calibrate(train['x']), 'z_te': calibrate(study['test']['x']),
            'y': train['y'], 'traj': train['traj'], 'eval_times': study['eval_times']}
