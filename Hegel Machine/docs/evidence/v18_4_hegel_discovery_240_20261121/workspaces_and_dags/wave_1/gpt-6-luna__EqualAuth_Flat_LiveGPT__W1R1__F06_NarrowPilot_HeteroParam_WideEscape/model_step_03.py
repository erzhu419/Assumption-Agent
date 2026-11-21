import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def predict_params(p):
        A, k = math.exp(p[0]), math.exp(p[1])
        b = p[2]
        return A * u / (k + u) * np.exp(np.clip(b * (1.0 - s), -20, 20))
    p0 = np.array([math.log(3.0), math.log(0.8), 0.7])
    res = least_squares(lambda p: predict_params(p) - y, p0, max_nfev=20000)
    return tuple(res.x)

def predict(row):
    p = fit(_rows)
    A, k, b = math.exp(p[0]), math.exp(p[1]), p[2]
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return float(A * u / (k + u) * math.exp(max(-20.0, min(20.0, b * (1.0 - s)))))
