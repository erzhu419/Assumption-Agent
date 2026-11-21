import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def pred(p):
        vmax, rho, tau_fast, amp_adapt, tau_slow = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base * (1.0 - np.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - np.exp(-h / tau_slow))

    lower = np.array([0.01, 0.0, 0.01, 0.0, 0.05])
    upper = np.array([30.0, 0.999, 100.0, 10.0, 200.0])
    starts = [
        [6.5, 0.70, 1.0, 0.5, 12.0],
        [5.0, 0.5, 0.3, 0.3, 30.0],
        [9.0, 0.8, 2.0, 1.0, 8.0],
        [4.0, 0.3, 0.1, 0.2, 60.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(lambda p: pred(p) - y, x0=np.clip(x0, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=20000)
        loss = float(np.sum(result.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    return best[1]

def predict(row):
    p = fit(_FIT_ROWS)
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

_FIT_ROWS = []
