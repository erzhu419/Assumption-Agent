import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None


def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float) / TAU_SCALE
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    inhibition = s / (1.0 + s)
    root_u = np.sqrt(u)

    def residual(p):
        j0, beta_bar, rho, c_sqrt = p
        return j0 * u * np.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * inhibition) + c_sqrt * root_u - y

    bounds = ([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf])
    best = min((least_squares(residual, [3.0, beta, 0.70, 0.0], bounds=bounds, x_scale='jac')
                for beta in (0.3, 1.0, 3.0)), key=lambda result: result.cost)
    _params = best.x


def predict(row) -> float:
    j0, beta_bar, rho, c_sqrt = _params
    u = row['u_now'] / TAU_SCALE
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u))