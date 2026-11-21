import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    dose = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    impact = float(row['impact_idx']) * float(row['a_foot'])
    return dose, impact

def fit(train_rows):
    global _PARAMS
    doses = np.array([_features(r)[0] for r in train_rows], dtype=float)
    impacts = np.array([_features(r)[1] for r in train_rows], dtype=float)
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    # y = baseline + gain*dose/(half_saturation+dose) - impact_loss*impact
    def residual(p):
        baseline, gain, half_saturation, impact_loss = p
        pred = baseline + gain * doses / (half_saturation + doses) - impact_loss * impacts
        return pred - ys
    result = least_squares(residual, x0=np.array([0.0, 8.0, 1.0, 3.0]),
                           bounds=([-20.0, 0.0, 1e-8, 0.0], [20.0, 100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    baseline, gain, half_saturation, impact_loss = _PARAMS
    dose, impact = _features(row)
    return float(baseline + gain * dose / (half_saturation + dose) - impact_loss * impact)