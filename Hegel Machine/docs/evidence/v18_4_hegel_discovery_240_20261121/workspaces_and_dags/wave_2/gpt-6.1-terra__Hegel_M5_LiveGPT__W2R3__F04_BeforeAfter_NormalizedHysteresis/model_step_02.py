import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = [r for r in train_rows if abs(r['u_now'] - r['u_past']) > 1e-12]
    h = np.array([r['H'] for r in rows], dtype=float)
    z = np.array([(r['y'] - r['x_now']) / (r['u_now'] - r['u_past']) for r in rows], dtype=float)

    def residual(p):
        a, b, slow, gap = p
        return a * np.exp(-slow * h) + b * np.exp(-(slow + gap) * h) - z

    best = None
    for slow in (0.05, 0.5, 1.5):
        for gap in (0.5, 2.0, 5.0):
            result = least_squares(residual, [200.0, -700.0, slow, gap], bounds=([-10000, -10000, 0, 0.0001], [10000, 10000, 20, 30]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    a, b, slow, gap = _params
    h = row['H']
    response = a * np.exp(-slow * h) + b * np.exp(-(slow + gap) * h)
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * response)
