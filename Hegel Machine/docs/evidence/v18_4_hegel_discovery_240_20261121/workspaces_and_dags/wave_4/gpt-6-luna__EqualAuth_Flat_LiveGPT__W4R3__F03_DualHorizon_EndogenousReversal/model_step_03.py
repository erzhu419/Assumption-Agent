import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)

    def predict_params(p):
        amp_fast, amp_slow = p[0], p[1]
        tau_fast = math.exp(p[2])
        tau_slow = tau_fast + math.exp(p[3])
        return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

    def residual(p):
        return predict_params(p) - y

    result = least_squares(residual, x0=np.array([1.5, 1.5, math.log(0.5), math.log(8.0)]),
                           bounds=([0.0, 0.0, math.log(1e-4), math.log(1e-4)],
                                   [20.0, 20.0, math.log(1e3), math.log(1e4)]),
                           max_nfev=50000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, amp_slow = float(_params[0]), float(_params[1])
    tau_fast = math.exp(float(_params[2]))
    tau_slow = tau_fast + math.exp(float(_params[3]))
    u, h = float(row['u_now']), float(row['H'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))