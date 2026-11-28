import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _pairs(x):
    return x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]]


def _calibrate(u, v, x, transform, start, lower, upper):
    def residual(params):
        d = _aff(transform(u, params))
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, start, bounds=(lower, upper), max_nfev=140).x
    d = _aff(transform(u, p))
    return _aff(transform(x, p)) @ _ls(d, v)


def _components(s, w):
    b, c = s['panel_b'], s['panel_c']
    affine = _ls(_aff(b['u']), b['v'])
    u = (w - affine[0]) @ np.linalg.pinv(affine[1:])
    def wave(x, p):
        return x + p[1] * np.sin(p[0] * x)
    def convex(x, p):
        return x + p[1] * (np.logaddexp(p[0]*x, -p[0]*x) - np.log(2.0))
    v = _calibrate(b['u'], b['v'], u, wave, [2.9,-.4], [1.,-2.], [6.,0.])
    q = _calibrate(c['u'], c['v2'], u, convex, [1.1,.91], [.25,0.], [3.,4.])
    coef = _ls(_aff(_pairs(c['u'])), c['v1'])
    return u, v, q, coef


def _kernel_predictions(v, q, targets, config):
    from scipy.linalg import solve
    from scipy.spatial.distance import cdist

    n = len(targets)
    features = np.column_stack((v, _pairs(v), q))
    scaled = (features - features[:n].mean(axis=0)) / np.maximum(features[:n].std(axis=0), 1e-8)
    linear = scaled @ scaled[:n].T / scaled.shape[1]
    pq = scaled[:, v.shape[1]:]
    distance = cdist(pq, pq[:n], 'sqeuclidean') / pq.shape[1]
    predictions = []
    for kind, rate, weight, ridge in config:
        if kind == 'pq':
            nonlinear = np.exp(-rate * distance)
        else:
            nonlinear = (1.0 + linear)**2
        kernel = 1.0 + linear + weight * nonlinear
        coefficients = solve(kernel[:n] + ridge*np.eye(n), targets, assume_a='pos')
        predictions.append(kernel[n:] @ coefficients)
    return np.stack(predictions)


_CONFIG = (
    ('pq', .08, .30, .00001),
    ('pq', .08, 3., .0003),
    ('pq', .16, 1., .00003),
    ('pq', .25, 20., .00001),
    ('pq', .25, 20., .003),
    ('quad', 0., 1., .015),
    ('pq', .12, 3., .00003),
    ('pq', .25, 1., .0001),
    ('pq', .10, 1., .0001),
)
_Y_WEIGHTS = (0.0, 0.0, 0.5164360648, 0.0, 0.0756475511, 0.407916384, 0.0, 0.0, 0.0)
_T_WEIGHTS = (0.0, 0.1214696665, 0.0, 0.0, 0.0, 0.4584866805, 0.0, 0.2778066778, 0.1422369752)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    w = np.asarray(inputs['node_00']['w'])
    _, v, q, _ = _components(study, w)
    train = study['train']
    targets = np.column_stack((train['y'], train['traj']))
    p = _kernel_predictions(v, q, targets, _CONFIG)
    y = np.asarray(_Y_WEIGHTS) @ p[:, :, 0]
    t = np.tensordot(_T_WEIGHTS, p[:, :, 1:], axes=1)
    tail = t[:, -1]
    return {'y_hat': y, 'state_pred': t, 'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
