import numpy as np
from scipy.optimize import least_squares


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _panel_a(panel):
    d, v = _affine(panel['u']), panel['v']
    w = np.linalg.lstsq(d, v, rcond=None)[0]
    for _ in range(20):
        r = np.linalg.norm(d @ w - v, axis=1)
        scale = max(np.median(r), 1e-9)
        q = 1 / np.hypot(1, r / scale)
        w = np.linalg.lstsq(d * q[:, None], v * q[:, None], rcond=None)[0]
    return w


def _panel_b(panel):
    u, v = panel['u'], panel['v']

    def basis(x, params):
        amplitude, frequency = params
        return _affine(x - amplitude * np.sin(frequency * x))

    def residual(params):
        d = basis(u, params)
        return (d @ np.linalg.lstsq(d, v, rcond=None)[0] - v).ravel()

    params = least_squares(residual, [0.4, 2.87],
                           bounds=([0.0, 1.5], [0.8, 4.5])).x
    w = np.linalg.lstsq(basis(u, params), v, rcond=None)[0]
    return lambda x: basis(x, params) @ w


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = _panel_a(study['panel_a'])
    b = _panel_b(study['panel_b'])

    def coordinates(x):
        return b(np.linalg.solve(a[1:].T, (x - a[0]).T).T)

    train, test = study['train'], study['test']
    return {'z_tr': coordinates(train['x']), 'z_te': coordinates(test['x']),
            'train_y': train['y'], 'train_traj': train['traj'],
            'eval_times': study['eval_times']}
