import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    indoor = float(row['u_indoor'])
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    area = float(row['a_foot'])
    impact = float(row['impact_idx'])
    return indoor, outdoor, area, impact

def fit(train_rows):
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # A compact mechanistic form: activity provides benefit; impact-area
    # burden suppresses the resulting vitality multiplicatively.
    def pred(p, x):
        indoor, outdoor, area, impact = x.T
        benefit = np.maximum(1e-9, p[0] + p[1]*indoor + p[2]*outdoor)
        burden = impact * area
        return benefit * np.exp(-p[3]*burden - p[4]*burden*burden)
    starts = [np.array([0.5, 1.0, 1.0, 1.0, 0.1]),
              np.array([1.0, 1.0, 1.0, 2.0, 0.0])]
    best = None
    for start in starts:
        res = least_squares(lambda p: pred(p, X)-y, start,
                            bounds=([-100,-100,-100,-20,-20],[100,100,100,20,20]),
                            max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = _COEFFICIENTS
    indoor, outdoor, area, impact = _features(row)
    burden = impact * area
    benefit = max(1e-9, p[0] + p[1]*indoor + p[2]*outdoor)
    return float(benefit * math.exp(-p[3]*burden - p[4]*burden*burden))
