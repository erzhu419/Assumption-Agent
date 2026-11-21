import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit y = u_now * [Af*(1-exp(-H/tf)) - As*(1-exp(-H/ts))].
    # Log parameters enforce positive amplitudes and time constants; bounds
    # keep the fast and slow scales in their physically ordered regimes.
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)

    def pred(p):
        af, tf, ass, ts = np.exp(p)
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))

    def residual(p):
        return pred(p) - y

    lower = np.log([1e-5, 1e-3, 1e-5, 1.01])
    upper = np.log([20.0, 2.0, 20.0, 200.0])
    best = None
    # Multistart numerical fitting addresses the strong short-horizon
    # amplitude/time-constant correlation.
    for tf0 in (0.05, 0.2, 0.8):
        for ts0 in (5.0, 20.0, 80.0):
            p0 = np.log([2.0, tf0, 1.0, ts0])
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=10000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, result.x)
    params = np.exp(best[1])

    def predict(row):
        af, tf, ass, ts = params
        H = float(row['H'])
        return float(row['u_now']) * (af * (-math.expm1(-H / tf)) - ass * (-math.expm1(-H / ts)))
    return predict

def predict(row):
    return fit([])(row)
