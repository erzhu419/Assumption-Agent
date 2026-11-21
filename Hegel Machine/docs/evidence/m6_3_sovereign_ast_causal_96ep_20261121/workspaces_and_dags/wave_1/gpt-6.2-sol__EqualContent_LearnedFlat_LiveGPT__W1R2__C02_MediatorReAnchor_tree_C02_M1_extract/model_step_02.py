import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    m = np.array([r['m_2'] for r in rows], dtype=float)
    raw = np.sqrt(np.array([r['u_raw'] for r in rows], dtype=float))
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        vmax, k, c = p
        return vmax * m / (k + m) + c * raw - y

    vmax, k, c = least_squares(residual, [8.3, 1.35, 0.32], bounds=([0.0, 1e-9, -np.inf], np.inf)).x

    def predict(row) -> float:
        mediator = float(row['m_2'])
        return float(vmax * mediator / (k + mediator) + c * math.sqrt(float(row['u_raw'])))

    return predict
