import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _adjacent(x):
    return x * np.roll(x, -1, axis=1)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel_a, panel_c = study['panel_a'], study['panel_c']
    a = _affine(panel_a['u'])
    v = panel_a['v']
    beta = _fit(a, v)
    # Estimate the forward map before inversion so response outliers do not
    # become high-leverage calibration inputs.
    for _ in range(12):
        residual = np.linalg.norm(v - a @ beta, axis=1)
        scale = max(3 * np.median(residual), 1e-6)
        root_weight = (1 + (residual / scale) ** 2) ** -0.5
        beta = _fit(a * root_weight[:, None], v * root_weight[:, None])
    residual = np.linalg.norm(v - a @ beta, axis=1)
    inlier = residual < max(8 * np.median(residual), 1e-4)
    beta = _fit(a[inlier], v[inlier])

    x = np.concatenate((study['train']['x'], study['test']['x']))
    z = np.linalg.solve(beta[1:].T, (x - beta[0]).T).T
    u = panel_c['u']
    c = _affine(_adjacent(z)) @ _fit(_affine(_adjacent(u)), panel_c['v1'])
    return {'u': u, 'z': z, 'v2': panel_c['v2'], 'c': c}
