import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))

def fit(train_rows):
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=[8.0, 0.4, 0.5], bounds=([0.0, -3.0, -10.0], [50.0, 3.0, 10.0]))
    predict.params = result.x.copy()
    return predict.params

def predict(row) -> float:
    return float(_value(row, predict.params))