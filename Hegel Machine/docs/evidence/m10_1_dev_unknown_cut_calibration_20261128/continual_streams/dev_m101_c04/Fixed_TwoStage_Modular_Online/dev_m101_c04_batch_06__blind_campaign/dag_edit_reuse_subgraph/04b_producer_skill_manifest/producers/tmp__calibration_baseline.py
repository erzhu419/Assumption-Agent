import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    n = np.roll(u, -1, axis=1)
    designs = np.stack((np.ones_like(u), u, n, u*n, u**3), axis=-1)
    weights = np.ones(len(u))
    for _ in range(10):
        w = np.sqrt(weights)
        coef = np.array([np.linalg.lstsq(designs[:, j]*w[:, None], v[:, j]*w, rcond=None)[0] for j in range(u.shape[1])])
        r = np.linalg.norm(np.einsum('nck,ck->nc', designs, coef)-v, axis=1)
        scale = max(np.median(r), 1e-8)
        weights = 1/(1+(r/(2*scale))**2)
    keep = np.argsort(r)[:-3]
    return np.array([np.linalg.lstsq(designs[keep, j], v[keep, j], rcond=None)[0] for j in range(u.shape[1])]).T


def _inverse(x, c):
    bias, linear, cross, product, cubic = c
    def forward(z):
        n = np.roll(z, -1)
        return bias + linear*z + cross*n + product*z*n + cubic*z**3
    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/linear, gtol=1e-11).x for row in x])


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    c = _calibrate(s['panel_a'])
    b = s['panel_b']
    offset, *linear = np.linalg.lstsq(_affine(b['u']), b['v'], rcond=None)[0]
    linear = np.asarray(linear)
    def coords(x):
        return np.linalg.solve(linear.T, (_inverse(x, c)-offset).T).T
    train = s['train']
    return {'z_tr': coords(train['x']), 'z_te': coords(s['test']['x']), 'y': train['y'], 'traj': train['traj']}
