import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    outdoor = float(row['u_outdoor'])
    weather = float(row['p_weather'])
    footprint = float(row['a_foot'])
    impact = float(row['impact_idx'])
    space_factor = 1.0 / (1.0 + math.exp(6.0 * (footprint - 1.5)))
    effective_dose = outdoor * weather + indoor * space_factor
    impact_load = (indoor * impact) ** 2
    return effective_dose, impact_load

def fit(train_rows):
    global _params
    xs = [_features(row) for row in train_rows]
    ys = np.asarray([float(row['y']) for row in train_rows], dtype=float)

    def residual(theta):
        g_root, k_e, c_imp = theta
        return np.asarray([
            g_root * dose / (k_e + dose) - c_imp * load
            for dose, load in xs
        ]) - ys

    result = least_squares(
        residual,
        x0=np.array([8.0, 1.0, 1.0]),
        bounds=([1e-8, 1e-8, 0.0], [100.0, 100.0, 100.0])
    )
    _params = result.x
    return predict

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    dose, load = _features(row)
    g_root, k_e, c_imp = _params
    return float(g_root * dose / (k_e + dose) - c_imp * load)