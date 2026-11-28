import numpy as np
from scipy.optimize import least_squares


def _fit_forward(panel):
    u, v = panel['u'], panel['v']
    affine = np.column_stack((np.ones(len(u)), u))
    weights = np.ones(len(u))
    for _ in range(30):
        coeff = np.linalg.lstsq(affine * weights[:, None], v * weights[:, None], rcond=None)[0]
        residual = np.linalg.norm(affine @ coeff - v, axis=1)
        scale = max(2 * np.median(residual), 1e-12)
        weights = 1 / np.sqrt(1 + (residual / scale) ** 2)

    # The three corrupted reference responses are row outliers.
    keep = np.argsort(residual)[:-3]
    a = u[keep]
    b = np.roll(a, -1, axis=1)
    eye = np.broadcast_to(np.eye(a.shape[1]), (len(a), a.shape[1], a.shape[1]))
    design = np.concatenate((eye, a[..., None] * eye,
                             b[..., None], (a * b)[..., None], (a ** 3)[..., None]), axis=2)
    p = np.linalg.lstsq(design.reshape(a.size, -1), v[keep].ravel(), rcond=None)[0]
    n = a.shape[1]
    return p[:n], p[n:2*n], p[-3], p[-2], p[-1]


def _forward(a, params):
    offset, linear, neighbor, product, cubic = params
    b = np.roll(a, -1, axis=-1)
    return offset + linear*a + neighbor*b + product*a*b + cubic*a**3


def _invert(observed, params):
    offset, linear, neighbor, product, cubic = params
    i = np.arange(len(linear))
    j = (i + 1) % len(i)
    affine = np.diag(linear)
    affine[i, j] = neighbor
    initial = np.linalg.solve(affine, (observed - offset).T).T

    def jacobian(a):
        jac = np.diag(linear + product*a[j] + 3*cubic*a*a)
        jac[i, j] = neighbor + product*a
        return jac

    return np.stack([least_squares(lambda a: _forward(a, params) - row, guess,
                                   jac=jacobian, gtol=1e-12).x
                     for row, guess in zip(observed, initial)])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    observed = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    params = _fit_forward(study['panel_a'])
    return {'calibrated_x': _invert(observed, params), 'panel_b': study['panel_b']}
