import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def fit(train_rows):
    global _P
    u = np.array([float(r['u_now']) / 0.1 for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def predict_params(q):
        vmax, k_half, alpha_inh, c_leak = np.exp(q)
        return vmax * (u ** 2 / (k_half + u ** 2)) / (1.0 + alpha_inh * s) - c_leak * u

    best = None
    for start in ([5.0, 2.0, 0.3, 0.05], [8.0, 10.0, 1.0, 0.01], [3.0, 0.5, 0.05, 0.2]):
        q0 = np.log(np.maximum(start, 1e-8))
        result = least_squares(lambda q: predict_params(q) - y, q0, max_nfev=20000)
        loss = float(np.sum(result.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, np.exp(result.x))
    _P = best[1]
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, k_half, alpha_inh, c_leak = _P
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return float(vmax * (u ** 2 / (k_half + u ** 2)) / (1.0 + alpha_inh * s) - c_leak * u)
