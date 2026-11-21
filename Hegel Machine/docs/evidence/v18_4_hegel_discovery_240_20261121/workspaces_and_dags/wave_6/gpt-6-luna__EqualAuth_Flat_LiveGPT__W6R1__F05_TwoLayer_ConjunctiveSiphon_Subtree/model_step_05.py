import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    u = np.asarray([max(1e-8, float(r['u_now'])) for r in train_rows], dtype=float)
    s = np.asarray([max(0.0, float(r['s_grad'])) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)

    def predict_params(p, uu, ss):
        vmax, km, rho, c = p
        return vmax * (1.0 - rho * ss / (1.0 + ss)) * uu / (km + uu) + c * np.sqrt(uu)

    scale = max(1.0, float(np.max(np.abs(y))))
    best = None
    for start in ([5.0, 0.01, 0.5, 5.0], [10.0, 0.02, 0.8, 10.0], [3.0, 0.005, 0.2, 2.0]):
        res = least_squares(lambda p: (predict_params(p, u, s) - y) / scale,
                            x0=np.asarray(start, dtype=float),
                            bounds=([0.0, 1e-7, 0.0, 0.0], [1e4, 10.0, 0.999, 1e4]),
                            max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    global _params
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, km, rho, c = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * math.sqrt(u))