import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')

def _features(row):
    o, i, p, a, d = (float(row[k]) for k in _KEYS)
    # Center/scale around the observed deployment regime.
    x = np.array([o / 3.0, i / 3.0, (a - 1.5) / 0.5, d, (o / 3.0) * d, (i / 3.0) * d,
                  (i / 3.0) * ((a - 1.5) / 0.5), (i / 3.0) * d * ((a - 1.5) / 0.5),
                  (i / 3.0) ** 2, d ** 2], dtype=float)
    return np.r_[1.0, x]

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    # Mild regularization stabilizes the underdetermined quadratic surface.
    lam = 0.03
    def residual(beta):
        return np.r_[X @ beta - y, math.sqrt(lam) * beta[1:]]
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    return result.x.tolist()

def predict(row):
    beta = np.asarray(fit(_TRAIN_ROWS), dtype=float)
    return float(_features(row) @ beta)
