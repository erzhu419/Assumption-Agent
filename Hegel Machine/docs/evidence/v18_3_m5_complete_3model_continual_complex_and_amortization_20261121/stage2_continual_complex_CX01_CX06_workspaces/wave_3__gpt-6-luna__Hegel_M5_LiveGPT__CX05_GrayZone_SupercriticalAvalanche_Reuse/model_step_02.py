import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def model(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                + c_loc * np.sqrt(u)
                - k_aval * np.maximum(0.0, s - s_crit) * u)
    lower = np.array([0.01, 0.0, -5.0, 0.0, 0.0])
    upper = np.array([30.0, 0.999, 5.0, 10.0, 3.0])
    starts = [np.array([6.5, 0.70, 0.1, 1.0, q]) for q in (1.2, 1.4, 1.6, 1.8)]
    best = None
    for p0 in starts:
        result = least_squares(lambda p: model(p) - y, p0, bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = fit_params
    s = float(row['s_grad'])
    u = float(row['u_now'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                 + c_loc * np.sqrt(u)
                 - k_aval * max(0.0, s - s_crit) * u)
