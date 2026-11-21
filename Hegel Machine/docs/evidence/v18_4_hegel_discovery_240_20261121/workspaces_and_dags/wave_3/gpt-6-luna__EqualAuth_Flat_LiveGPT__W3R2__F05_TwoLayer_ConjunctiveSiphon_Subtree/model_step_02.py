import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def features(row):
        u = max(1e-8, float(row['u_now']))
        s = max(0.0, float(row['s_grad']))
        # Concave saturating base response, with a gradient-dependent sink.
        return np.array([u/(1.0+u), math.sqrt(u), s*u/(1.0+s), s*s*u/(1.0+s*s)])
    X = np.vstack([features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # bounded least squares; baseline positive, sink coefficients nonnegative
    def residual(p):
        return X @ np.array([p[0], p[1], -p[2], -p[3]]) - y
    result = least_squares(residual, x0=np.array([3.0, 1.0, 0.2, 0.1]), bounds=([-100,-100,0,0],[100,100,100,100]))
    fit._params = np.array([result.x[0], result.x[1], -result.x[2], -result.x[3]])
    return fit._params

def predict(row):
    p = getattr(fit, '_params', np.array([3.0, 1.0, -0.2, -0.1]))
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    x = np.array([u/(1.0+u), math.sqrt(u), s*u/(1.0+s), s*s*u/(1.0+s*s)])
    return float(x @ p)
