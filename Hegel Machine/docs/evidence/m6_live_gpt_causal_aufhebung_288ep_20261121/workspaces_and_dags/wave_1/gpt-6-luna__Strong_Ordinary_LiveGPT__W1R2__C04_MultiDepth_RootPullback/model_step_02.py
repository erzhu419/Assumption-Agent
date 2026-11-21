import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _dose(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    af = float(row['a_foot'])
    impact = float(row['impact_idx'])
    # Stable evaluation of the documented logistic footprint penalty.
    z = 6.0 * (af - 1.5)
    if z >= 0:
        indoor_factor = math.exp(-z) / (1.0 + math.exp(-z))
    else:
        indoor_factor = 1.0 / (1.0 + math.exp(z))
    return uo * pw + ui * indoor_factor, (ui * impact) ** 2

def fit(train_rows):
    global _params
    doses = np.array([_dose(r) for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        g, k, c = p
        e, loss_basis = doses[:, 0], doses[:, 1]
        return g * e / (k + e) - c * loss_basis - ys
    result = least_squares(residual, x0=np.array([6.0, 1.0, 1.0]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first.')
    g, k, c = _params
    e, loss_basis = _dose(row)
    return float(g * e / (k + e) - c * loss_basis)
