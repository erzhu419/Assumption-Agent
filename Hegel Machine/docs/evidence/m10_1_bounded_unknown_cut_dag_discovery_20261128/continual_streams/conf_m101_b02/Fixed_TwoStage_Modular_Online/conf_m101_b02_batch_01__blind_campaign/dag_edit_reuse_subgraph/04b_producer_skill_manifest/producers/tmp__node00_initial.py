import numpy as np


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    d = np.column_stack((np.ones(len(u)), u))
    w = np.ones(len(u))
    for _ in range(12):
        coef = np.linalg.lstsq(d * w[:, None], v * w[:, None], rcond=None)[0]
        r = np.linalg.norm(v - d @ coef, axis=1)
        w = 1.0 / np.sqrt(1 + (r / max(np.median(r), 0.001)) ** 2)
    return coef[0], coef[1:]


def _panel_fit(u, v, z, terms):
    def design(x):
        return np.column_stack((np.ones(len(x)), x, *terms(x)))
    return design(z) @ np.linalg.lstsq(design(u), v, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    x = np.vstack((train['x'], test['x']))
    offset, gain = _calibrate(study['panel_a'])
    z = np.linalg.solve(gain.T, (x - offset).T).T
    b, c = study['panel_b'], study['panel_c']
    fd = _panel_fit(b['u'], b['v'], z, lambda u: (np.sin(2.815 * u),))
    c1 = _panel_fit(c['u'], c['v1'], z, lambda u: (np.tanh(2.25 * u),))
    c2 = _panel_fit(c['u'], c['v2'], z, lambda u: (u ** 2, np.cos(2.5 * u)))
    cd = np.column_stack((c1, c2))
    g = np.column_stack((np.ones(len(z)), z, fd, cd))
    return {'g': g, 'fd': fd, 'cd': cd, 'y': train['y'], 'traj': train['traj']}
