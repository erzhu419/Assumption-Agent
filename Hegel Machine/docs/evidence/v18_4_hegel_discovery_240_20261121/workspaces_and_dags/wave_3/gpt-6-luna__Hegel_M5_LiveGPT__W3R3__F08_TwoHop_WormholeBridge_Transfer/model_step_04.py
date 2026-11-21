import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    un = u  # supplied characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * np.sqrt(un)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    predict.params = result.x
    return predict.params

def predict(row) -> float:
    return float(_value(row, predict.params))