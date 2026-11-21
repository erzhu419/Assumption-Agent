import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(0.0, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, uu, ss):
        j0, beta, rho, c = p
        return j0 * uu * np.exp(-beta / (0.5 + uu)) * (1.0 - rho * ss / (1.0 + ss)) + c * np.sqrt(uu)
    result = least_squares(lambda p: pred(p, u, s) - y,
                           x0=np.array([8.0, 0.5, 0.5, 0.1]),
                           bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 10.0, 1.0, np.inf]),
                           max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row])
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    j0, beta, rho, c = p
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))