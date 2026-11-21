import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _eval(kind, p, u, s):
    if kind == 'siphon_sqrt':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * np.sqrt(u)
    if kind == 'siphon_leak':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) - p[2] * u
    if kind == 'hill':
        return p[0] * u*u / (p[1] + u*u) / (1.0 + p[2]*s) - p[3]*u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1.0 - p[2]*s/(1.0+s)) + p[3]*np.sqrt(u)

def fit(train_rows):
    global _params, _kind
    u = np.array [float(r['u_now']) for r in train_rows]
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    candidates = [
        ('siphon_sqrt', [10., 0.7, 1.], [0., 0., -20.], [100., 2., 20.]),
        ('siphon_leak', [10., 0.7, 0.5], [0., 0., 0.], [100., 2., 20.]),
        ('hill', [10., 1., 0.5, 0.5], [0., 0.0001, 0., 0.], [100., 100., 20., 20.]),
        ('barrier', [5., 1., 0.7, 1.], [0., 0., 0., -20.], [100., 20., 2., 20.])
    ]
    best = float('inf')
    for kind, start, lower, upper in candidates:
        result = least_squares(lambda p: _eval(kind, p, u, s) - y, start, bounds=(lower, upper), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = float(np.mean(result.fun ** 2))
        if error < best:
            best, _kind, _params = error, kind, result.x

def predict(row):
    return float(_eval(_kind, _params, float(row['u_now']), float(row['s_grad'])))