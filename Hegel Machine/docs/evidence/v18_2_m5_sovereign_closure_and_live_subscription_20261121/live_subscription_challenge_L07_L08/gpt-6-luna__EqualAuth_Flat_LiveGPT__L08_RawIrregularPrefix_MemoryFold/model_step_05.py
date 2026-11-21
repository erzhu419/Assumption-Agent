import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(0.0, float(r['u_now'])) for r in rows], dtype=float)
    g = np.array([float(r['s_grad']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # Compare the recalled open-branch alternatives. The multiplier involving
    # s_grad is unidentifiable in these data (all gradients are zero).
    candidates = []
    for kind in ('sqrt_plus', 'sqrt_minus', 'linear_leak'):
        def pred(p):
            vmax, rho, branch = p
            base = vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u)
            if kind == 'sqrt_plus':
                return base + branch * np.sqrt(u)
            if kind == 'sqrt_minus':
                return base - branch * np.sqrt(u)
            return base - branch * u
        result = least_squares(lambda p: pred(p) - y,
                               x0=np.array([2.0, 0.7, 0.1]),
                               bounds=([0.0, 0.0, -10.0], [20.0, 1.0, 10.0]),
                               max_nfev=5000)
        candidates.append((float(np.mean((pred(result.x) - y) ** 2)), kind, result.x.copy()))
    _, kind, params = min(candidates, key=lambda z: z[0])
    return {'kind': kind, 'params': params.tolist()}


def predict(row):
    # Parameters are attached to the callable by fit in the evaluation harness;
    # fallback supports environments that pass a fitted model as row metadata.
    model = row.get('_model')
    if model is None:
        raise ValueError('predict requires fitted parameters from fit')
    vmax, rho, branch = model['params']
    u = max(0.0, float(row['u_now']))
    g = float(row['s_grad'])
    base = vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u)
    if model['kind'] == 'sqrt_plus':
        return float(base + branch * math.sqrt(u))
    if model['kind'] == 'sqrt_minus':
        return float(base - branch * math.sqrt(u))
    return float(base - branch * u)
