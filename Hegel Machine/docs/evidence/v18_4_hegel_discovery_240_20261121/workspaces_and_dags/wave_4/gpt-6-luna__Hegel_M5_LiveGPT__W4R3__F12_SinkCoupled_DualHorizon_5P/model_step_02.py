import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([r['y'] for r in rows], dtype=float)

    def predict_params(r, p):
        vmax, rho, amp, log_fast, log_gap = p
        tau_fast = math.exp(log_fast)
        tau_slow = tau_fast + math.exp(log_gap)
        H = float(r['H'])
        u = float(r['u_now'])
        s = float(r['s_grad'])
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base * (1.0 - math.exp(-H / tau_fast)) - amp * u * (1.0 - math.exp(-H / tau_slow))

    def residuals(p):
        return np.array([predict_params(r, p) for r in rows]) - ys

    best = None
    for start in ([6.5, 0.70, 1.0, math.log(0.5), math.log(5.0)],
                  [5.0, 0.50, 1.5, math.log(1.0), math.log(10.0)],
                  [8.0, 0.85, 0.8, math.log(0.2), math.log(3.0)]):
        result = least_squares(
            residuals, np.array(start, dtype=float),
            bounds=([0.01, 0.0, 0.0, math.log(0.01), math.log(0.01)],
                    [30.0, 0.999, 20.0, math.log(20.0), math.log(100.0)]),
            max_nfev=20000
        )
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    p = best[1]
    return {'params': [float(v) for v in p]}

def predict(row):
    p = _MODEL['params']
    vmax, rho, amp, log_fast, log_gap = p
    tau_fast = math.exp(log_fast)
    tau_slow = tau_fast + math.exp(log_gap)
    H = float(row['H'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return float(base * (1.0 - math.exp(-H / tau_fast)) - amp * u * (1.0 - math.exp(-H / tau_slow)))

_MODEL = {'params': [6.5, 0.7, 1.0, math.log(0.5), math.log(5.0)]}
_fit = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _fit(train_rows)
    return _MODEL
