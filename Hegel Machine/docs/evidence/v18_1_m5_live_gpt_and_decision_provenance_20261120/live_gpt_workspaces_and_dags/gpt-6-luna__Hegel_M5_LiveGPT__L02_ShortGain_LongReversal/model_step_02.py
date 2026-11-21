import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    H = np.array([float(r['H']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    ratio = y / np.where(np.abs(u) > 1e-12, u, 1e-12)

    def unpack(p):
        amp_fast, amp_slow = p[0], p[1]
        tau_fast = math.exp(float(np.clip(p[2], -12.0, 12.0)))
        tau_slow = tau_fast + math.exp(float(np.clip(p[3], -12.0, 12.0)))
        return amp_fast, amp_slow, tau_fast, tau_slow

    def pred_ratio(p, h):
        af, ass, tf, ts = unpack(p)
        return af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts))

    best = None
    for tf0 in (0.15, 0.5, 1.0, 2.0):
        for ts_gap0 in (2.0, 10.0, 25.0):
            p0 = np.array([1.3, 1.8, math.log(tf0), math.log(ts_gap0)])
            res = least_squares(lambda p: pred_ratio(p, H) - ratio, p0, max_nfev=30000)
            score = float(np.sum(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, res.x.copy())
    return unpack(best[1])

def predict(row):
    af, ass, tf, ts = fit(_TRAIN_ROWS)
    h = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))