import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-4, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def model(p):
        vmax, rho, km, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u)) + c_loc * np.sqrt(u)

    result = least_squares(
        lambda p: model(p) - y,
        x0=np.array([5.0, 0.2, 1.0, 0.5]),
        bounds=([0.0, 0.0, 1e-6, 0.0], [100.0, 1.0, 100.0, 100.0]),
        max_nfev=20000,
    )
    return result.x


def predict(row):
    vmax, rho, km, c_loc = fit(_FIT_ROWS)
    u = max(1e-4, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u)) + c_loc * math.sqrt(u))