import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([float(r['H']) for r in train_rows])
    d = np.array([float(r['u_now']) - float(r['u_past']) for r in train_rows])
    x = np.array([float(r['x_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        a, b, k, l = p
        return x + d * (a * np.exp(-k * h) + b * np.exp(-l * h)) - y

    best = None
    for k, l in [(0.05, 1.0), (0.2, 2.0), (0.5, 3.0), (1.0, 4.0), (0.01, 0.8), (1.0, 1.5)]:
        z = np.column_stack((d * np.exp(-k * h), d * np.exp(-l * h)))
        ab = np.linalg.lstsq(z, y - x, rcond=None)[0]
        result = least_squares(residual, [ab[0], ab[1], k, l], bounds=([-1e6, -1e6, 0, 0], [1e6, 1e6, 30, 30]), max_nfev=5000)
        loss = np.dot(result.fun, result.fun)
        if best is None or loss < best[0]:
            best = (loss, result.x)
    _params = best[1]

def predict(row):
    a, b, k, l = _params
    h = float(row['H'])
    d = float(row['u_now']) - float(row['u_past'])
    return float(row['x_now'] + d * (a * np.exp(-k * h) + b * np.exp(-l * h)))