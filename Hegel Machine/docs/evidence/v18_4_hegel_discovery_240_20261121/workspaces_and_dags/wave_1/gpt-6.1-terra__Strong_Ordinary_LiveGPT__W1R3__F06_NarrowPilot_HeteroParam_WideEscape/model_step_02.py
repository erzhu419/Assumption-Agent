import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _evaluate(name, p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    if name == 'siphon_sqrt':
        vmax, rho, branch = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + branch * math.sqrt(max(0.0, u))
    if name == 'siphon_leak':
        vmax, rho, leak = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) - leak * u
    if name == 'hill':
        vmax, k, alpha, leak = p
        return vmax * u * u / (k + u * u) / (1.0 + alpha * s) - leak * u
    j0, beta, rho, branch = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + branch * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    specs = [
        ('siphon_sqrt', [6.0, 0.7, 0.5], ([0.0, 0.0, -20.0], [100.0, 2.0, 20.0])),
        ('siphon_leak', [6.0, 0.7, 0.2], ([0.0, 0.0, -20.0], [100.0, 2.0, 20.0])),
        ('hill', [6.0, 1.0, 0.5, 0.1], ([0.0, 0.00001, 0.0, -20.0], [100.0, 100.0, 100.0, 20.0])),
        ('barrier', [2.0, 1.0, 0.7, 0.5], ([0.0, 0.0, 0.0, -20.0], [100.0, 30.0, 2.0, 20.0]))
    ]
    best = float('inf')
    for name, start, bounds in specs:
        def residual(p):
            return np.array([_evaluate(name, p, r) - float(r['y']) for r in rows])
        result = least_squares(residual, start, bounds=bounds, max_nfev=20000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = float(np.dot(result.fun, result.fun))
        if error < best:
            best = error
            _model, _params = name, result.x
    return _params

def predict(row):
    return float(_evaluate(_model, _params, row))