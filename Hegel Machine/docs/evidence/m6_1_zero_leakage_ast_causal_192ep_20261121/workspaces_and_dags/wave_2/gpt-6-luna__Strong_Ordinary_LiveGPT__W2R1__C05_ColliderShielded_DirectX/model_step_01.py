import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return x / (1.1 + x), u

def fit(train_rows):
    rows = list(train_rows)
    A = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return A @ p - y
    result = least_squares(residual, x0=np.array([7.0, 0.5]))
    vmax, c_ctrl = result.x
    def predict(row):
        f, u = _features(row)
        return float(vmax * f + c_ctrl * u)
    return predict

def predict(row):
    # Fallback prior parameters from the direct-actuator historical mechanism.
    f, u = _features(row)
    return float(7.0 * f + 0.5 * u)
