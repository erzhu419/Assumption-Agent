import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)

    def base(r, vmax, rho):
        u = float(r['u_now'])
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))

    candidates = []
    # Compare the two documented square-root branch signs and the mass-drift leak.
    for kind in ('sqrt_plus', 'sqrt_minus', 'leak'):
        def pred(r, p):
            vmax, rho, c = p
            b = base(r, vmax, rho)
            u = float(r['u_now'])
            if kind == 'sqrt_plus':
                return b + c * math.sqrt(max(u, 0.0))
            if kind == 'sqrt_minus':
                return b - c * math.sqrt(max(u, 0.0))
            return b - c * u

        def residual(p):
            return np.array([pred(r, p) for r in rows]) - y

        result = least_squares(
            residual,
            x0=np.array([4.0, 0.3, 0.2]),
            bounds=(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.999, 100.0])),
            max_nfev=20000
        )
        candidates.append((float(np.dot(result.fun, result.fun)), kind, result.x))

    _, kind, p = min(candidates, key=lambda item: item[0])
    return {'kind': kind, 'params': np.asarray(p, dtype=float)}

def predict(row):
    model = fit(_TRAIN_ROWS)
    vmax, rho, c = model['params']
    u = float(row['u_now'])
    s = float(row['s_grad'])
    value = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if model['kind'] == 'sqrt_plus':
        value += c * math.sqrt(max(u, 0.0))
    elif model['kind'] == 'sqrt_minus':
        value -= c * math.sqrt(max(u, 0.0))
    else:
        value -= c * u
    return float(value)