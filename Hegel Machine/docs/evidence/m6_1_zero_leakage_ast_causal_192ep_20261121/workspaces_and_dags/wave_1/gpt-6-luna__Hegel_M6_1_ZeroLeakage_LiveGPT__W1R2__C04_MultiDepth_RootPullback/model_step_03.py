import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    area = float(row['a_foot'])
    impact = float(row['impact_idx'])
    # Stable evaluation of the compact-apartment space sigmoid.
    z = 6.0 * (area - 1.5)
    indoor = ui / (1.0 + math.exp(max(-700.0, min(700.0, z))))
    effective = uo * pw + indoor
    penalty_basis = (ui * impact) ** 2
    return effective, penalty_basis

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    X = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(theta):
        gain, half_sat, impact_coeff = theta
        return gain * X[:, 0] / (half_sat + X[:, 0]) - impact_coeff * X[:, 1] - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.8]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    gain, half_sat, impact_coeff = _PARAMS
    effective, penalty_basis = _features(row)
    return float(gain * effective / (half_sat + effective) - impact_coeff * penalty_basis)
