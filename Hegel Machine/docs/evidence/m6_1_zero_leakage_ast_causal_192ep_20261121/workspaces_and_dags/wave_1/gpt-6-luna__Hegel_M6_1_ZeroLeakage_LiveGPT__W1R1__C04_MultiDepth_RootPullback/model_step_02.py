import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = np.array([1.0, 1.0, 1.0])

def _predict_with(row, params):
    g_root, k_e, c_imp = params
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    effective_dose = outdoor + indoor
    impact_loss = c_imp * (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return g_root * effective_dose / (k_e + effective_dose) - impact_loss

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(params):
        return np.asarray([_predict_with(r, params) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.5]), bounds=(np.array([0.0, 1e-9, 0.0]), np.array([100.0, 100.0, 100.0])))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    return float(_predict_with(row, _PARAMS))