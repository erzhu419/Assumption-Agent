import numpy as np
from scipy.optimize import least_squares

def _features(row):
    # Weather-weight outdoor activity; allow indoor/outdoor benefits and
    # nonlinear footprint/impact losses to vary across the observed regime.
    o = float(row.get('u_outdoor', 0.0)) * float(row.get('p_weather', 0.0))
    i = float(row.get('u_indoor', 0.0))
    a = float(row.get('a_foot', 0.0))
    m = float(row.get('impact_idx', 0.0))
    z = np.array([o, i, a, m], dtype=float)
    return np.array([1.0, *z, o*o, i*i, a*a, m*m, o*i, o*a, o*m, i*a, i*m, a*m], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Mild coefficient regularization stabilizes this small pilot while keeping
    # the intercept and main effects data-driven.
    penalty = np.diag([0.0] + [0.03] * (X.shape[1] - 1))
    def residual(beta):
        return np.concatenate([X @ beta - y, penalty @ beta])
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    return result.x

def predict(row):
    return float(_features(row) @ _MODEL_COEFFICIENTS)
