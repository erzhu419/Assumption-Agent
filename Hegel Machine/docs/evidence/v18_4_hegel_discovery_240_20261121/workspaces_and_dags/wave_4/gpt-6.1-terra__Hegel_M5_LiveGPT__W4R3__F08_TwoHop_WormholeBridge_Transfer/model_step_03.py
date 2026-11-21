import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    a, b, c, d, e = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return a + b * np.log(u + c) - d * np.log(s + e)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = [ -100.0, -100.0, max(1e-8, -min(float(r['u_now']) for r in rows) + 1e-6), -100.0, max(1e-8, -min(float(r['s_grad']) for r in rows) + 1e-6)]
    upper = [100.0, 100.0, 100.0, 100.0, 100.0]
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    fits = []
    for c in (0.05, 0.5, 2.0):
        for e in (0.05, 0.5, 2.0):
            p0 = [3.0, 1.5, max(c, lower[2] + 1e-5), 1.0, max(e, lower[4] + 1e-5)]
            fits.append(least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000))
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x

def predict(row):
    return float(_value(_params, row))