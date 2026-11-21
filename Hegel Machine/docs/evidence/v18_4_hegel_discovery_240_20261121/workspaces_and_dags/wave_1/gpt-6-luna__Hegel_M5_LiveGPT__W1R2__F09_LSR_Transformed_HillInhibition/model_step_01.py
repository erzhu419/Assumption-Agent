import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(p, r) for r in rows]) - y
    starts = [
        [6.0, 1.0, 0.5, 0.1],
        [8.0, 0.5, 1.0, 0.3],
        [5.0, 2.0, 0.1, 0.0],
        [10.0, 3.0, 2.0, 0.5],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=([0.0, 1e-8, 0.0, 0.0], [30.0, 100.0, 100.0, 10.0]), max_nfev=20000)
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    return {'params': best[1]}

def predict(row):
    return float(_predict_params(fit._params, row))