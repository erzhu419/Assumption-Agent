import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _evaluate(kind, p, u, s):
    if kind == 'siphon_sqrt':
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u)
    if kind == 'siphon_leak':
        vmax, rho, leak = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) - leak * u
    if kind == 'hill':
        vmax, k, alpha, leak = p
        return vmax * u**2 / (k + u**2) / (1.0 + alpha * s) - leak * u
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _model, _params
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    specs = [
        ('siphon_sqrt', [10.0, 0.7, 1.0], ([0.0, 0.0, -20.0], [100.0, 2.0, 20.0])),
        ('siphon_leak', [10.0, 0.7, 0.1], ([0.0, 0.0, 0.0], [100.0, 2.0, 20.0])),
        ('hill', [10.0, 1.0, 0.2, 0.1], ([0.0, 0.00001, 0.0, 0.0], [100.0, 100.0, 20.0, 20.0])),
        ('barrier', [3.0, 1.0, 0.7, 1.0], ([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 2.0, 20.0]))
    ]
    best = float('inf')
    for kind, initial, bounds in specs:
        result = least_squares(lambda p: _evaluate(kind, p, u, s) - y, initial, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = float(np.mean(result.fun ** 2))
        if error < best:
            best, _model, _params = error, kind, result.x

def predict(row):
    return float(_evaluate(_model, _params, float(row['u_now']), float(row['s_grad'])))