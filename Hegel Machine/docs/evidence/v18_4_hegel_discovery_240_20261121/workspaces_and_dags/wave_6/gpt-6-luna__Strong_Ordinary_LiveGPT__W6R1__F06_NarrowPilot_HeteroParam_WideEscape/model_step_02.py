import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit the three-parameter 2-layer siphon law with a signed sqrt branch term.
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, ss):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * ss / (1.0 + ss)) * (uu / (1.2 + uu)) + c_loc * np.sqrt(np.maximum(uu, 0.0))
    scale = max(float(np.std(y)), 1e-6)
    res = least_squares(lambda p: (pred(p, u, s) - y) / scale,
                        x0=np.array([6.0, 0.8, 0.5]), max_nfev=20000)
    return res.x.tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(p[0] * (1.0 - p[1] * row['s_grad'] / (1.0 + row['s_grad'])) * (row['u_now'] / (1.2 + row['u_now'])) + p[2] * math.sqrt(max(row['u_now'], 0.0)))