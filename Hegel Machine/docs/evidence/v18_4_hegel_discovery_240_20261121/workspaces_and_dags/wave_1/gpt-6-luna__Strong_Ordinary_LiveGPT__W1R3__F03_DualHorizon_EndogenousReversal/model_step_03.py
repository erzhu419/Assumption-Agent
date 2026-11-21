import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    h = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Optimize log-parameters to keep amplitudes and timescales positive.
    def residual(logp):
        af, ass, tf, ts = np.exp(logp)
        pred = u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
        return pred - y
    best = None
    for guess in ([1.5, 1.5, 0.5, 15.0], [2.0, 2.0, 1.0, 30.0], [1.0, 1.5, 0.2, 8.0], [3.0, 3.0, 2.0, 50.0]):
        result = least_squares(residual, np.log(guess), bounds=(np.log([1e-6]*4), np.log([1e3]*4)), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, np.exp(result.x))
    return {'amp_fast': float(best[1][0]), 'amp_slow': float(best[1][1]), 'tau_fast': float(best[1][2]), 'tau_slow': float(best[1][3])}

def predict(row):
    p = fit(_TRAIN_ROWS)
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (p['amp_fast'] * (-math.expm1(-h / p['tau_fast'])) - p['amp_slow'] * (-math.expm1(-h / p['tau_slow']))))