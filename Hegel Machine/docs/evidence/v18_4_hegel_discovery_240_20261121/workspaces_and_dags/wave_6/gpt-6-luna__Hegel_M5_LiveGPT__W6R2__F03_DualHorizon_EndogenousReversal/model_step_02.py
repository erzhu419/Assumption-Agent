import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    def residual(p):
        return pred(p) - y
    best = None
    for tf0, ts0 in [(0.2, 10.0), (0.5, 20.0), (1.0, 30.0), (0.1, 5.0)]:
        p0 = [1.2, 1.5, tf0, ts0]
        res = least_squares(residual, p0, bounds=([0.0, 0.0, 1e-4, 1e-4], [100.0, 100.0, 100.0, 1000.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    af, ass, tf, ts = best.x
    return {'amp_fast': float(af), 'amp_slow': float(ass), 'tau_fast': float(tf), 'tau_slow': float(ts)}

def predict(row):
    p = fit(globals().get('_train_rows', []))
    H = float(row['H']); u = float(row['u_now'])
    return u * (p['amp_fast'] * (1.0 - math.exp(-H / p['tau_fast'])) - p['amp_slow'] * (1.0 - math.exp(-H / p['tau_slow'])))