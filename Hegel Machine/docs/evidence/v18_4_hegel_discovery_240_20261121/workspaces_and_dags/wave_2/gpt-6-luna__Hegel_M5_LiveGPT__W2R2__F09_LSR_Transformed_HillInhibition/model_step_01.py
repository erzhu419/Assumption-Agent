import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def predict_params(p, u_now, s_grad):
        vmax, k_half, alpha_inh, c_leak = p
        u_n = float(u_now)  # tau_scale = 1.0
        return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * float(s_grad)) - c_leak * u_n
    def residuals(p):
        return np.array([predict_params(p, ui, si) for ui, si in zip(u, s)]) - y
    result = least_squares(residuals, x0=np.array([4.0, 1.0, 0.5, 0.1]), bounds=(np.zeros(4), np.full(4, np.inf)))
    return {'params': result.x}

def predict(row):
    vmax, k_half, alpha_inh, c_leak = fit_params['params']
    u_n = float(row['u_now'])  # tau_scale = 1.0
    return float(vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * float(row['s_grad'])) - c_leak * u_n)