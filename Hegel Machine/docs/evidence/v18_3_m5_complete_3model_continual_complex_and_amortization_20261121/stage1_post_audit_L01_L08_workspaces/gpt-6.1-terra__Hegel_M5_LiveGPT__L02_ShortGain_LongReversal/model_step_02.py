import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(logp):
        af, tf, a_s, ts = np.exp(logp)
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))) - y

    lower = np.log([0.001, 0.01, 0.001, 0.5])
    upper = np.log([100.0, 10.0, 100.0, 200.0])
    best = None
    for tf in (0.2, 0.7, 2.0):
        for ts in (3.0, 10.0, 40.0):
            for amp in (1.0, 3.0):
                start = np.log([amp, tf, amp, ts])
                result = least_squares(residual, start, bounds=(lower, upper), max_nfev=5000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
                if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                    best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, a_s, ts = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / ts))))
