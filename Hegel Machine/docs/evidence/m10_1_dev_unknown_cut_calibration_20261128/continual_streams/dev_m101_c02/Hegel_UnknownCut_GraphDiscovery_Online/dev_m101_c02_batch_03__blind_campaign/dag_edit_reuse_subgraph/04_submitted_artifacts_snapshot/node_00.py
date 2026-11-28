import numpy as np
from scipy.optimize import least_squares


def _fit_forward(u, v):
    n, d = u.shape
    nxt = np.roll(u, -1, axis=1)
    axis = np.eye(d)
    # Axis-specific offsets and gains; the three nonlinear/coupled terms
    # share coefficients around the coordinate cycle.
    design = np.concatenate((
        np.broadcast_to(axis, (n, d, d)),
        u[:, :, None] * axis,
        np.stack((nxt, u * nxt, u**3), axis=-1),
    ), axis=-1)
    weight = np.ones(n)
    for _ in range(40):
        w = np.sqrt(weight)[:, None, None]
        coef = np.linalg.lstsq((design * w).reshape(n*d, -1),
                               (v * w[..., 0]).ravel(), rcond=None)[0]
        residual = np.linalg.norm(design @ coef - v, axis=1)
        scale = max(1e-10, 0.1 * np.median(residual))
        weight = scale / np.maximum(scale, residual)
    # Panel A contains three grossly corrupted observations.
    good = np.argsort(residual)[:-3]
    return np.linalg.lstsq(design[good].reshape(len(good)*d, -1),
                           v[good].ravel(), rcond=None)[0]


def _invert(x, coef):
    d = x.shape[1]
    j = np.arange(d)
    k = (j + 1) % d
    bias, gain = coef[:d], coef[d:2*d]
    neighbor, pair, cubic = coef[2*d:]

    def forward(u):
        return bias + gain*u + (neighbor + pair*u)*u[k] + cubic*u**3

    def jacobian(u):
        jac = np.diag(gain + pair*u[k] + 3*cubic*u**2)
        jac[j, k] = neighbor + pair*u
        return jac

    initial = np.linalg.solve(jacobian(np.zeros(d)), (x-bias).T).T
    result = np.empty_like(initial)
    for i, (row, start) in enumerate(zip(x, initial)):
        result[i] = least_squares(lambda u: forward(u) - row, start,
                                  jac=jacobian, gtol=1e-12).x
    return result


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    coef = _fit_forward(panel['u'], panel['v'])
    n_train = len(study['train']['x'])
    calibrated = _invert(np.concatenate((study['train']['x'], study['test']['x'])), coef)
    return {'u_tr': calibrated[:n_train], 'u_te': calibrated[n_train:],
            'panel_b': study['panel_b']}
