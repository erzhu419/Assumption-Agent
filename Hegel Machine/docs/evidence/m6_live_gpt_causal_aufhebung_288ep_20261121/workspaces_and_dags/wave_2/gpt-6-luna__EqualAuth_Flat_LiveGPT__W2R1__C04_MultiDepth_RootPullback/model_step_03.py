import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    outdoor = float(row.get('u_outdoor', 0.0))
    indoor = float(row.get('u_indoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    foot = float(row.get('a_foot', 1.0))
    impact = float(row.get('impact_idx', 0.0))
    # Low-weather deployment: indoor dose is gated by available footprint and
    # discounted by biomechanical impact loss; retain the small outdoor pathway.
    dose = indoor * foot * (1.0 - impact)
    return np.array([1.0, dose, dose * dose, outdoor * weather, impact], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Mild shrinkage stabilizes the weakly identified residual terms.
    def residual(b):
        return np.r_[X @ b - y, 0.02 * b[2:],]
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _PARAMS)
