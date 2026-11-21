import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    indoor = row['u_indoor'] / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
    effective_dose = row['u_outdoor'] * row['p_weather'] + indoor
    impact_loss_basis = (row['u_indoor'] * row['impact_idx']) ** 2
    return effective_dose, impact_loss_basis

def fit(train_rows):
    y = np.array([r['y'] for r in train_rows], dtype=float)
    features = [_features(r) for r in train_rows]
    def residual(params):
        g_root, k_e, c_imp = params
        pred = np.array([g_root * (dose / (k_e + dose)) - c_imp * loss
                         for dose, loss in features])
        return pred - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5]),
                           bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    g_root, k_e, c_imp = fit(_TRAIN_ROWS)
    dose, loss = _features(row)
    return float(g_root * (dose / (k_e + dose)) - c_imp * loss)
