import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[r['u_outdoor']*r['p_weather'] + r['u_indoor'], r['a_foot'], r['impact_idx']] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, x):
        g, k, loss, power = p
        dose, foot, impact = x
        return g * dose / (k + dose + 1e-12) - loss * impact * (foot ** power)
    def resid(p):
        return np.array([pred(p, x) for x in X]) - y
    best = None
    for p0 in ([7,1,5,1], [10,2,4,2], [5,0.5,3,0.5]):
        res = least_squares(resid, p0, bounds=([0,1e-5,0,0],[50,50,50,5]), max_nfev=10000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2): best = res
    return best.x

def predict(row):
    g, k, loss, power = _params
    dose = row['u_outdoor']*row['p_weather'] + row['u_indoor']
    return float(g*dose/(k+dose+1e-12) - loss*row['impact_idx']*(row['a_foot']**power))

_params = [7.0, 1.0, 5.0, 1.0]
# fit() returns parameters; call site should retain them for prediction.
