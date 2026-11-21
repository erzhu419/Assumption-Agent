import math
import numpy as np
from scipy.optimize import least_squares

def _predict_with(row, params):
    g_root, k_e, c_imp = params
    e_eff = float(row['u_outdoor']) * float(row['p_weather'])
    e_eff += float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    loss = c_imp * (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return g_root * e_eff / (k_e + e_eff) - loss

def fit(train_rows):
    def residuals(params):
        return np.asarray([_predict_with(r, params) - float(r['y']) for r in train_rows])
    result = least_squares(residuals, x0=np.array([6.0, 0.8, 1.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    return tuple(float(v) for v in result.x)

def predict(row):
    params = fit(_TRAIN_ROWS)
    return float(_predict_with(row, params))
