import math
import numpy as np
from scipy.optimize import least_squares


def _response(u, s, p):
    vmax, km, rho, local = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + local * math.sqrt(u)


def fit(train_rows):
    global _params
    rows = [(float(r['u_now']) / 0.05, float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residual(p):
        return np.array([_response(u, s, p) - y for u, s, y in rows])
    _params = least_squares(residual, [7.0, 1.0, 0.6, -0.3], bounds=([0.0, 1e-8, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x


def predict(row) -> float:
    return float(_response(float(row['u_now']) / 0.05, float(row['s_grad']), _params))