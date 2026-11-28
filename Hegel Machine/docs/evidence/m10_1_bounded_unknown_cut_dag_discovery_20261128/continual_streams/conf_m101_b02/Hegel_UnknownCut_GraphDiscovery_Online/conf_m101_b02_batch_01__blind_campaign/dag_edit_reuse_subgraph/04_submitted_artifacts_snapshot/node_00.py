import numpy as np
from scipy.optimize import least_squares


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _warp(x, k, c):
    return x * (1 + c * (-np.expm1(-(k*x)**2)))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    a, b = s['panel_a'], s['panel_b']
    A = _design(a['u'])
    v = a['v']
    p = np.linalg.lstsq(A, v, rcond=None)[0]
    for _ in range(16):
        w = 1 / np.sqrt(1 + (np.linalg.norm(A @ p-v, axis=1)/.04)**2)
        p = np.linalg.lstsq(A*w[:, None], v*w[:, None], rcond=None)[0]
    r = np.linalg.norm(A @ p-v, axis=1)
    good = r < max(.01, 8*np.median(r))
    if good.sum() >= A.shape[1]:
        p = np.linalg.lstsq(A[good], v[good], rcond=None)[0]
    x = np.concatenate((s['train']['x'], s['test']['x']))
    z = np.linalg.solve(p[1:].T, (x-p[0]).T).T
    u, v = b['u'], b['v']
    def profile(q):
        m = _design(_warp(u, *q))
        fit = np.linalg.lstsq(m, v, rcond=None)[0]
        return m, fit
    def residual(q):
        m, fit = profile(q)
        return (m @ fit-v).ravel()
    q = least_squares(residual, [1., -10.], bounds=([.3, -50.], [2., -.5])).x
    m, fit = profile(q)
    return {'z': z, 'b': _design(_warp(z, *q)) @ fit,
            'calibration_error': np.array([np.sqrt(np.mean((A[good] @ p-a['v'][good])**2)), np.sqrt(np.mean((m @ fit-v)**2))])}
