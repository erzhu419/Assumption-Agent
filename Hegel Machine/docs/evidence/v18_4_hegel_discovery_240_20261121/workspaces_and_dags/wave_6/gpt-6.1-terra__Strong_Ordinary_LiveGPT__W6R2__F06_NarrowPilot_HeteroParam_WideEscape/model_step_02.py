import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(name, p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    if name == 'siphon_sqrt':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * np.sqrt(max(u, 0.0))
    if name == 'siphon_linear':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) - p[2] * u
    if name == 'hill':
        return p[0] * u*u / (p[1] + u*u) / (1.0 + p[2]*s) - p[3]*u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1.0 - p[2]*s/(1.0+s)) + p[3]*np.sqrt(max(u, 0.0))

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    specs = [
        ('siphon_sqrt', [10., 0.7, 1.], ([0., 0., -100.], [1000., 2., 100.])),
        ('siphon_linear', [10., 0.7, 0.1], ([0., 0., 0.], [1000., 2., 100.])),
        ('hill', [10., 1., 0.5, 0.1], ([0., 0.00001, 0., 0.], [1000., 1000., 100., 100.])),
        ('barrier', [5., 1., 0.7, 1.], ([0., 0., 0., -100.], [1000., 100., 2., 100.]))
    ]
    best = float('inf')
    for name, initial, bounds in specs:
        result = least_squares(lambda p: np.array([_value(name, p, r) for r in rows]) - y, initial, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        err = float(np.dot(result.fun, result.fun))
        if err < best:
            best, _model, _params = err, name, result.x
    return None

def predict(row):
    return float(_value(_model, _params, row))