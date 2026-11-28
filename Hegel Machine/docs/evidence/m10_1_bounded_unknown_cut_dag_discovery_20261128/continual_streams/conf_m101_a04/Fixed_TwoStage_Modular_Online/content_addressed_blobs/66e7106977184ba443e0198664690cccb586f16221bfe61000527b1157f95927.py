import numpy as np


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _fit_a(panel):
    a, v = _affine(panel['u']), panel['v']
    coef = np.linalg.lstsq(a, v, rcond=None)[0]
    # Fit in the forward direction; one weight covers all channels of a row.
    for _ in range(12):
        residual = np.linalg.norm(v - a @ coef, axis=1)
        scale = max(2.5 * np.median(residual), 1e-9)
        weight = 1 / np.hypot(1, residual / scale)
        coef = np.linalg.lstsq(a * weight[:, None], v * weight[:, None], rcond=None)[0]
    return coef


def _v1_basis(u):
    a, b, c, d = u.T
    return np.column_stack((np.ones(len(u)), a*b, a*d, b*c, c*d))


def _v2_basis(u):
    return np.column_stack((_affine(u), np.cos(2.4*u)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    a = _fit_a(study['panel_a'])
    u = np.linalg.solve(a[1:].T, (x - a[0]).T).T
    c = study['panel_c']
    v1 = np.linalg.lstsq(_v1_basis(c['u']), c['v1'], rcond=None)[0]
    v2 = np.linalg.lstsq(_v2_basis(c['u']), c['v2'], rcond=None)[0]
    # The frozen quadratic readout acts on the two panel C responses.
    h = np.column_stack((_v1_basis(u) @ v1, _v2_basis(u) @ v2))
    return {'h': h, 'n_train': len(train['x']), 'y': train['y'], 'traj': train['traj']}
