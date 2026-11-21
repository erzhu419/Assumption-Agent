import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 3.0

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    # Guided rho and branch bounds; multiple initializations reduce local-minimum risk.
    starts = [np.array([6.5, b, 0.70, c], dtype=float) for b in (0.2, 1.0, 2.0, 4.0) for c in (-0.3, 0.3)]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.0, 0.0, 0.05, -2.0], [30.0, 15.0, 0.95, 2.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return float(_predict_params(row, fit_params))