import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def features(rho):
        sink = 1.0 - rho * s / (1.0 + s)
        return [u, u * sink, u / (1.2 + u) * sink, np.sqrt(u), np.ones_like(u)]

    best = None
    for fast_kind in range(5):
        for slow_kind in range(5):
            def residual(p):
                vmax, rho, local, af, tf, asl, ts = p
                f = features(rho)
                base = vmax * f[2] + local * f[3]
                return base + af * f[fast_kind] * (-np.expm1(-h / tf)) - asl * f[slow_kind] * (-np.expm1(-h / ts)) - y
            for tf, ts in [(0.7, 12.0), (2.0, 20.0)]:
                p0 = [2.0, 0.7, 0.2, 1.0, tf, 1.0, ts]
                result = least_squares(residual, p0, bounds=([-20, 0, -20, -20, 0.03, -20, 1.0], [20, 2, 20, 20, 10, 20, 100]), max_nfev=1500)
                error = float(np.mean(result.fun ** 2))
                if best is None or error < best[0]:
                    best = (error, fast_kind, slow_kind, result.x.copy())
    _model = best[1:]

def predict(row):
    fk, sk, p = _model
    vmax, rho, local, af, tf, asl, ts = p
    u, s, h = float(row['u_now']), float(row['s_grad']), float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    f = [u, u * sink, u / (1.2 + u) * sink, math.sqrt(u), 1.0]
    return float(vmax * f[2] + local * f[3] + af * f[fk] * (-math.expm1(-h / tf)) - asl * f[sk] * (-math.expm1(-h / ts)))