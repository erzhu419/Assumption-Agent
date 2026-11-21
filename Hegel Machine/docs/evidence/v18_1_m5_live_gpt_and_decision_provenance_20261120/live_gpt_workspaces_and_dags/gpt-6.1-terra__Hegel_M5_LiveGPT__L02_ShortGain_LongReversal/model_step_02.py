import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, slow_amp, tf, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - slow_amp * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.1, 0.3, 1.0, 2.0):
        for ts in (2.0, 8.0, 20.0, 50.0):
            p0 = np.log([2.0, 3.0, tf, ts])
            result = least_squares(residual, p0, bounds=(np.log([0.001, 0.001, 0.005, 0.05]), np.log([100.0, 100.0, 100.0, 1000.0])), max_nfev=5000)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, slow_amp, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - slow_amp * (-np.expm1(-h / ts))))