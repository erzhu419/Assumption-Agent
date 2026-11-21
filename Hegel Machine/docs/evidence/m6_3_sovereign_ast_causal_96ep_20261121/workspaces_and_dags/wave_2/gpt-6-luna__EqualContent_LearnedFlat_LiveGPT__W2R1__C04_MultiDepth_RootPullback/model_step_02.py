import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    # Rainy regime: indoor contribution is essential; footprint and impact
    # jointly moderate root vitality. Fit a compact nonlinear mechanistic form.
    return np.array([1.0, float(row['u_outdoor']), float(row['u_indoor']),
                     float(row['a_foot']), float(row['impact_idx'])], dtype=float)

def fit(train_rows):
    global _params
    X = np.array([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Smooth exponential response allows competing indoor/outdoor gains and
    # footprint/impact loss without assuming the sunny-climate mechanism.
    def pred(p, row):
        o, i, a, m = float(row['u_outdoor']), float(row['u_indoor']), float(row['a_foot']), float(row['impact_idx'])
        return p[0] + p[1]*o + p[2]*i + p[3]*a + p[4]*m + p[5]*i*m + p[6]*a*m + p[7]*i*a
    p0 = np.array([1., 0., 1., 0., -1., -1., -1., 0.])
    result = least_squares(lambda p: np.array([pred(p, r) for r in train_rows])-y, p0, max_nfev=20000)
    _params = result.x
    return predict

def predict(row):
    p = _params
    o, i, a, m = float(row['u_outdoor']), float(row['u_indoor']), float(row['a_foot']), float(row['impact_idx'])
    return float(p[0] + p[1]*o + p[2]*i + p[3]*a + p[4]*m + p[5]*i*m + p[6]*a*m + p[7]*i*a)
