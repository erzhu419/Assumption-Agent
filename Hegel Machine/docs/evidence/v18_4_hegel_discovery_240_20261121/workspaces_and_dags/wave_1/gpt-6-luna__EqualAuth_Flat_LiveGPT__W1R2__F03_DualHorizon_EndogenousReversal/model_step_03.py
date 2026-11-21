import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    a_fast, a_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (a_fast * (1.0 - math.exp(-h / tau_fast)) - a_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    h = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def unpack(z):
        tf = math.exp(float(z[2]))
        return (math.exp(float(z[0])), math.exp(float(z[1])), tf, tf + math.exp(float(z[3])))
    def residual(z):
        af, ass, tf, ts = unpack(z)
        pred = u * (af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts)))
        return pred - y
    best = None
    for start in ([math.log(1.5), math.log(0.7), math.log(1.0), math.log(8.0)], [math.log(2.0), math.log(0.8), math.log(0.5), math.log(15.0)], [math.log(1.0), math.log(0.5), math.log(2.0), math.log(20.0)]):
        result = least_squares(residual, np.array(start), max_nfev=30000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, unpack(result.x))
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))