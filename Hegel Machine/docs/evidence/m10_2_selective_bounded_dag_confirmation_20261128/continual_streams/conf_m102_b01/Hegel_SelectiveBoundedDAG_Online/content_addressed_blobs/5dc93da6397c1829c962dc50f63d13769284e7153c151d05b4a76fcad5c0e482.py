import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _a(u, v, x):
    j = np.roll(np.arange(u.shape[1]), -1)
    def basis(t, z):
        return np.stack([np.ones_like(t), t, z, t*z, t**3], axis=-1)
    f = basis(u, u[:, j])
    keep = np.max(np.abs(v-u), axis=1) < 3
    coef = np.stack([_fit(f[keep, i], v[keep, i]) for i in range(u.shape[1])])
    return np.stack([least_squares(lambda z: (basis(z, z[j])*coef).sum(1)-row, row, gtol=1e-11).x for row in x])


def _b(u, v, x):
    # A shared odd response acts on each coordinate before affine mixing.
    # Profile out the mixing and offset using only the calibration pairs.
    def basis(z, p):
        curvature, linear, correction, width = p
        z2 = z*z
        response = (z - z/np.sqrt(1 + curvature*z2)
                    + linear*z + correction*z/(1 + width*z2))
        return np.column_stack([np.ones(len(z)), response])

    def residual(p):
        f = basis(u, p)
        return (f @ _fit(f, v) - v).ravel()

    p = least_squares(residual, [1., 0., .1, 15.],
                      bounds=([.02, -1., -1., 1.], [10., 1., 1., 100.]),
                      max_nfev=400).x
    return basis(x, p) @ _fit(basis(u, p), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    x = np.concatenate([study['train']['x'], study['test']['x']])
    return {'w': _b(b['u'], b['v'], _a(a['u'], a['v'], x))}
