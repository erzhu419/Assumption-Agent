import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _fit_panel_a(panel):
    x, v = _affine(panel['u']), panel['v']
    coef = _fit(x, v)
    for _ in range(12):
        error = np.linalg.norm(v - x @ coef, axis=1)
        scale = max(np.median(error), 1e-12)
        weight = 1.0 / np.hypot(1.0, error / scale)
        coef = _fit(x * weight[:, None], v * weight[:, None])
    error = np.linalg.norm(v - x @ coef, axis=1)
    keep = np.argsort(error)[:-3]
    return _fit(x[keep], v[keep])


def _panel_b_design(x, params):
    amplitude, frequency = params
    return _affine(x - amplitude * np.sin(frequency * x))


def _fit_panel_b(panel):
    u, v = panel['u'], panel['v']

    def profile(params):
        x = _panel_b_design(u, params)
        coef = _fit(x, v)
        return (x @ coef - v).ravel(), coef

    params = np.array([0.4, 2.85])
    residual, coef = profile(params)
    delta = np.eye(2) * 1e-5
    for _ in range(30):
        jac = np.column_stack([(profile(params + d)[0] - profile(params - d)[0]) / 2e-5 for d in delta])
        step = _fit(jac, -residual)
        if np.linalg.norm(step) < 1e-8:
            break
        loss = residual @ residual
        for _ in range(12):
            trial = params + step
            error, trial_coef = profile(trial)
            if error @ error < loss:
                params, residual, coef = trial, error, trial_coef
                break
            step *= 0.5
        else:
            break
    return params, coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = _fit_panel_a(study['panel_a'])
    params, b = _fit_panel_b(study['panel_b'])

    def calibrate(x):
        u = np.linalg.solve(a[1:].T, (x - a[0]).T).T
        return _panel_b_design(u, params) @ b

    train, test = study['train'], study['test']
    return {
        'z': calibrate(train['x']),
        'z_test': calibrate(test['x']),
        'y': train['y'].copy(),
        'traj': train['traj'].copy(),
    }
