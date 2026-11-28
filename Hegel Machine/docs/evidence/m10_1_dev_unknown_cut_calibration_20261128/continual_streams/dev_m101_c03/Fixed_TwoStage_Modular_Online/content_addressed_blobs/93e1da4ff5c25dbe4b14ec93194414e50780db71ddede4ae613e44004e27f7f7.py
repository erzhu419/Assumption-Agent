import numpy as np


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _sensor_fit(panel):
    u, v = panel['u'], panel['v']
    x = np.stack((np.ones_like(u), u), axis=-1)

    def fit(w):
        return np.column_stack([_fit(x[:, j] * w[:, None], v[:, j] * w)
                                for j in range(u.shape[1])])

    w = np.ones(len(u))
    for _ in range(16):
        coef = fit(w)
        r = np.linalg.norm(v - coef[0] - u * coef[1], axis=1)
        scale = max(4e-4, 0.8 * np.median(r))
        w = 1 / np.sqrt(1 + (r / scale)**2)
    return fit(w)


def _v1(x):
    return np.sign(x) * -np.expm1(-1.8 * np.abs(x))


def _v2(x):
    return np.cos(2.4 * x)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    offset, gain = _sensor_fit(study['panel_a'])
    z = (np.vstack((train['x'], test['x'])) - offset) / gain
    b, c = study['panel_b'], study['panel_c']
    linear_sensor = _design(z) @ _fit(_design(b['u']), b['v'])
    first = _design(_v1(z)) @ _fit(_design(_v1(c['u'])), c['v1'])
    second = _design(_v2(z)) @ _fit(_design(_v2(c['u'])), c['v2'])
    mixed = (first[:, :, None] * second[:, None, :]).reshape(len(z), -1)
    linear = _design(np.column_stack((linear_sensor, second, mixed)))
    return {'n': np.asarray(len(train['x'])), 'linear': linear, 'mixed': mixed,
            'y': train['y'], 'traj': train['traj'], 'eval_times': study['eval_times']}
