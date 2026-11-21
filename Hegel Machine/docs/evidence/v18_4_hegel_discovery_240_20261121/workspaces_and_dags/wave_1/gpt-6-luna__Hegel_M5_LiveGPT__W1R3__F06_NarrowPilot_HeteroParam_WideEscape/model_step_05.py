import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    hill = u*u / (k_half + u*u)
    return vmax * hill / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([5.0, 1.0, 1.0, 0.1], [8.0, 4.0, 2.0, 0.0], [4.0, 0.3, 0.3, 0.3]):
        result = least_squares(residual, start, bounds=([1e-8, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return tuple(float(v) for v in best.x)

def predict(row):
    raise RuntimeError('Call fit(train_rows) and use its returned parameters with _predict_params(row, params).')