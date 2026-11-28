import numpy as np
from scipy.optimize import least_squares


def _ls(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _fit_a(u, v):
    neighbor = np.roll(u, -1, axis=1)
    designs = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=1)
    weights = np.ones(len(u))
    for _ in range(20):
        coef = np.array([_ls(d * weights[:, None], y * weights)
                         for d, y in zip(designs.transpose(2, 0, 1), v.T)])
        err = np.linalg.norm(v - np.einsum('nkj,jk->nj', designs, coef), axis=1)
        scale = max(1.5 * np.median(err), 1e-6)
        weights = 1 / np.hypot(1, err / scale)
    return coef


def _invert_a(x, coef):
    offset, linear, coupling, product, cubic = coef.T
    idx = np.arange(len(offset))
    nxt = np.roll(idx, -1)

    def residual(u, row):
        neighbor = u[nxt]
        return offset + linear*u + coupling*neighbor + product*u*neighbor + cubic*u**3 - row

    def jacobian(u):
        jac = np.diag(linear + product*u[nxt] + 3*cubic*u*u)
        jac[idx, nxt] = coupling + product*u
        return jac

    initial = np.linalg.solve(jacobian(np.zeros(len(offset))), (x - offset).T).T
    return np.array([least_squares(lambda u: residual(u, row), start, jac=jacobian,
                                  gtol=1e-12).x for row, start in zip(x, initial)])


def _harmonic_design(u, frequency, amplitude):
    return np.column_stack((np.ones(len(u)), u - amplitude * np.sin(frequency * u)))


def _fit_b(u, v):
    # Profile the affine mixing while fitting the shared coordinate warp.
    def residual(params):
        design = _harmonic_design(u, *params)
        return (design @ _ls(design, v) - v).ravel()

    params = least_squares(residual, [2.8, 0.4],
                           bounds=([2.0, 0.1], [4.0, 0.7])).x
    return params, _ls(_harmonic_design(u, *params), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    coef_a = _fit_a(a['u'], a['v'])
    params_b, coef_b = _fit_b(b['u'], b['v'])

    def calibrate(x):
        return _harmonic_design(_invert_a(x, coef_a), *params_b) @ coef_b

    train = study['train']
    return {'z_tr': calibrate(train['x']), 'z_te': calibrate(study['test']['x']),
            'y': train['y'], 'traj': train['traj'], 'eval_times': study['eval_times']}
