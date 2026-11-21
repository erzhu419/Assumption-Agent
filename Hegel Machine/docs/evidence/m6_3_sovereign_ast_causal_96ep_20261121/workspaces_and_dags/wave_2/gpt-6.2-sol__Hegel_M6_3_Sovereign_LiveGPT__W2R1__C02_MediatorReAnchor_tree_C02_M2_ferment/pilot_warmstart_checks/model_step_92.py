import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    m = np.array([float(r['m_2']) for r in train_rows])
    raw = np.sqrt(np.array([float(r['u_raw']) for r in train_rows]))
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        (vmax, km, c_raw) = p
        return vmax * m / (km + m) + c_raw * raw - y
    (vmax, km, c_raw) = least_squares(residual, [7.4, 1.32, 0.28], bounds=([0.0, 1e-08, -np.inf], [np.inf, np.inf, np.inf])).x

    def fitted_predict(row):
        mediator = float(row['m_2'])
        return float(vmax * mediator / (km + mediator) + c_raw * math.sqrt(float(row['u_raw'])))
    global _fitted_predict
    _fitted_predict = fitted_predict
    return fitted_predict

def predict(row) -> float:
    return _fitted_predict(row)