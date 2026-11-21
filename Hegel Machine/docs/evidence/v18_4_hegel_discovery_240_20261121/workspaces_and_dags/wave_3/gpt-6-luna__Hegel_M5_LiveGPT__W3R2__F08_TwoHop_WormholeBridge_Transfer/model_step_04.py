import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return np.array([u, u**4, g*u, g], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Numerically fit linear/quartic flux plus gradient-dependent bridge terms.
    def residual(p):
        return X @ p - y
    result = least_squares(residual, x0=np.array([1.0, -0.01, 0.0, 0.0]))
    return result.x

def predict(row):
    p = fit(_train_rows)
    return float(_features(row) @ p)