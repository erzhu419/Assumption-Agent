import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _value(p, row, kind):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    vmax, af, tf, slow, ts = p[:5]
    rho = p[5] if kind in (1, 3, 5, 7) else 0.7
    branch = p[6] * np.sqrt(max(u, 0.0)) if kind in (3, 7) else 0.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + branch
    fast = -np.expm1(-h / tf)
    delayed = -np.expm1(-h / ts)
    if kind < 4:
        return base + u * (af * fast - slow * delayed)
    return base * (1.0 + af * fast - slow * delayed)

def fit(train_rows):
    global _params, _kind
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for kind in (0, 1, 3, 4, 5, 7):
        n = 7 if kind in (3, 7) else (6 if kind in (1, 5) else 5)
        lo = [0.0, 0.0, 0.01, 0.0, 0.1] + ([0.0] if n >= 6 else []) + ([-10.0] if n == 7 else [])
        hi = [30.0, 30.0, 30.0, 30.0, 200.0] + ([2.0] if n >= 6 else []) + ([10.0] if n == 7 else [])
        def residual(p):
            return np.array([_value(p, r, kind) for r in rows]) - y
        local = None
        for tf, ts in ((0.4, 8.0), (1.5, 15.0), (3.0, 25.0), (0.15, 4.0)):
            initial = [4.0, 1.0, tf, 1.5, ts] + ([0.7] if n >= 6 else []) + ([0.0] if n == 7 else [])
            result = least_squares(residual, initial, bounds=(lo, hi), max_nfev=4000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            mse = float(np.mean(result.fun ** 2))
            if local is None or mse < local[0]:
                local = (mse, result.x)
        # Prefer the simpler mechanism when fits differ only at rounding scale.
        score = local[0] + (n - 5) * 1e-10
        if best is None or score < best[0]:
            best = (score, kind, local[1])
    _kind, _params = best[1], best[2]

def predict(row):
    return float(_value(_params, row, _kind))
