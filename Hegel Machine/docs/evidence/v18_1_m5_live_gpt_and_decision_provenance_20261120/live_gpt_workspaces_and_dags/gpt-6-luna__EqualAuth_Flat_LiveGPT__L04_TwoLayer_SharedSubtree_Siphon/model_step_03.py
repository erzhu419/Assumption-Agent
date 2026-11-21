import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def predict_p(p, r):
        vmax, rho, bx, bm = p
        u = max(float(r['u_now']), 0.0)
        s = float(r['s_grad'])
        core = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return core + bx * float(r['x_now']) + bm * float(r['mem_conv'])
    def residual(p):
        return np.array([predict_p(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.70, 0.0, 0.0]), bounds=([0.01, 0.0, -10.0, -10.0], [20.0, 1.0, 10.0, 10.0]))
    return result.x

def predict(row):
    p = fit(_train_rows)
    vmax, rho, bx, bm = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    core = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return float(core + bx * float(row['x_now']) + bm * float(row['mem_conv']))