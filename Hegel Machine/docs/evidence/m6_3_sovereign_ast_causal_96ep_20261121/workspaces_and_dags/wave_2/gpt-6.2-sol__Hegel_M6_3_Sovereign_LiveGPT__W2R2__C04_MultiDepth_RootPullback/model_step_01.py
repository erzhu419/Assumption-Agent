import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    indoor = float(row['u_indoor'])
    space = 1.0 / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(params):
        gain, half_sat, penalty = params
        return gain * dose / (half_sat + dose) - penalty * impact - y

    gain, half_sat, penalty = least_squares(residual, x0=[8.0, 0.8, 0.6], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf])).x

    def predict(row) -> float:
        d, i = _features(row)
        return float(gain * d / (half_sat + d) - penalty * i)

    return predict