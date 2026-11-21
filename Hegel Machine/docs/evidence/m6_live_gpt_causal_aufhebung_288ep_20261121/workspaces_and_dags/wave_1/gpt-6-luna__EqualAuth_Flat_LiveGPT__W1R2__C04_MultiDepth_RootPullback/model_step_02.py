import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row, k):
    indoor = float(row.get('u_indoor', 0.0))
    outdoor_weather = float(row.get('u_outdoor', 0.0)) * float(row.get('p_weather', 0.0))
    foot = float(row.get('a_foot', 1.0))
    impact = float(row.get('impact_idx', 0.0))
    space_dose = (indoor * indoor / (k + indoor)) * foot
    impact_loss = impact * foot
    return np.array([1.0, space_dose, impact_loss, outdoor_weather], dtype=float)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(z):
        k = math.exp(z[0])
        beta = z[1:]
        return np.array([_features(r, k) @ beta for r in train_rows]) - y
    best = None
    for k0 in (0.5, 1.5, 3.0):
        x0 = np.r_[math.log(k0), np.zeros(4)]
        sol = least_squares(residual, x0, max_nfev=5000)
        if best is None or np.dot(sol.fun, sol.fun) < np.dot(best.fun, best.fun):
            best = sol
    _PARAMS = (math.exp(best.x[0]), best.x[1:].copy())
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    k, beta = _PARAMS
    return float(_features(row, k) @ beta)
