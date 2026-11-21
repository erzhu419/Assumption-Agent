import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    af, ass, tf, ts = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))

def fit(train_rows):
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows], dtype=float)
    def unpack(z):
        return (z[0], z[1], math.exp(z[2]), math.exp(z[2]) + math.exp(z[3]))
    def residual(z):
        p = unpack(z)
        return np.asarray([_predict_params(r, p) for r in rows]) - ys
    # Positive amplitudes and ordered timescales (fast < slow).
    starts = [[1.5, 2.0, math.log(0.5), math.log(8.0)],
              [2.0, 2.5, math.log(0.8), math.log(15.0)],
              [1.0, 1.5, math.log(0.3), math.log(5.0)]]
    best = None
    for z0 in starts:
        result = least_squares(residual, z0,
            bounds=([0.0, 0.0, math.log(0.005), math.log(0.005)],
                    [20.0, 20.0, math.log(100.0), math.log(100.0)]),
            max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, unpack(result.x))
    return best[1]

def predict(row):
    return _predict_params(row, _fitted_params)
