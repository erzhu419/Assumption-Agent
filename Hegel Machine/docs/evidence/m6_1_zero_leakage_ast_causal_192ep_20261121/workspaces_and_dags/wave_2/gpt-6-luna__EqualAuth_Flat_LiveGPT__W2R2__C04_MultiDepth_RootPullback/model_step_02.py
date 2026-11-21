import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Rainy-regime mechanism: indoor activity contributes positively; outdoor
    # activity is weather-discounted; impact and footprint reduce vitality.
    X = []
    y = []
    for r in train_rows:
        indoor = float(r['u_indoor'])
        outdoor_eff = float(r['u_outdoor']) * float(r['p_weather'])
        impact = float(r['impact_idx'])
        foot = float(r['a_foot'])
        X.append([1.0, indoor, outdoor_eff, impact, foot, indoor * impact])
        y.append(float(r['y']))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    def residuals(beta):
        return X @ beta - y
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    result = least_squares(residuals, beta0)
    return result.x.tolist()

def predict(row):
    # Parameters are learned by fit; conventional train-then-predict API stores
    # them on the row-independent function object.
    beta = getattr(predict, '_beta', None)
    if beta is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    indoor = float(row['u_indoor'])
    outdoor_eff = float(row['u_outdoor']) * float(row['p_weather'])
    impact = float(row['impact_idx'])
    foot = float(row['a_foot'])
    return float(np.dot(beta, [1.0, indoor, outdoor_eff, impact, foot, indoor * impact]))

# Bind the fitted coefficients while preserving the required fit/predict API.
_fit_impl = fit
def fit(train_rows):
    beta = _fit_impl(train_rows)
    predict._beta = np.asarray(beta, dtype=float)
    return beta
