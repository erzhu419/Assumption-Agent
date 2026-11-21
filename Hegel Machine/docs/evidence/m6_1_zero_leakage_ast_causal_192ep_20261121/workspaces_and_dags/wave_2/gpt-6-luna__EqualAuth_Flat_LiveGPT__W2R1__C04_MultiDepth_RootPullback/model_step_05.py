import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    outdoor = float(row.get('u_outdoor', 0.0))
    indoor = float(row.get('u_indoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    foot = float(row.get('a_foot', 1.0))
    impact = float(row.get('impact_idx', 0.0))
    activity = math.log1p(max(0.0, indoor))
    weather_activity = weather * outdoor
    return np.array([1.0, activity, activity * activity, weather_activity,
                     impact * foot, impact * impact * foot,
                     activity * impact * foot, activity * foot], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    # Modest shrinkage stabilizes the small pilot while allowing impact/activity interaction.
    def residual(p):
        return np.concatenate([X @ p - y, 0.03 * p[1:]])
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=10000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _PARAMS)