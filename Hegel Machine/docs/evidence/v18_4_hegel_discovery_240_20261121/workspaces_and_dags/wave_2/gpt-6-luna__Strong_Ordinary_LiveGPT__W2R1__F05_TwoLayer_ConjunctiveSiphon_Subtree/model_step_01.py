import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    un = np.array([r['u_now'] / 0.05 for r in rows], dtype=float)
    sg = np.array([r['s_grad'] for r in rows], dtype=float)
    yy = np.array([r['y'] for r in rows], dtype=float)
    base = (un / (1.2 + un))

    candidates = []
    # The documented siphon branch allows either sign on the sqrt term;
    # mass-drift leakage is the alternative branch noted for this mechanism.
    for branch in ('sqrt_plus', 'sqrt_minus', 'leak'):
        def pred(p):
            vmax, rho, c = p
            core = vmax * (1.0 - rho * sg / (1.0 + sg)) * base
            if branch == 'sqrt_plus':
                return core + c * np.sqrt(un)
            if branch == 'sqrt_minus':
                return core - c * np.sqrt(un)
            return core - c * un
        result = least_squares(lambda p: pred(p) - yy,
                               x0=np.array([5.0, 0.5, 0.2]),
                               bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 100.0]),
                               max_nfev=10000)
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, branch, params = min(candidates, key=lambda z: z[0])
    return {'branch': branch, 'params': [float(v) for v in params]}

def predict(row):
    model = fit(_TRAIN_ROWS)
    vmax, rho, c = model['params']
    un = row['u_now'] / 0.05
    core = vmax * (1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])) * (un / (1.2 + un))
    if model['branch'] == 'sqrt_plus':
        return float(core + c * math.sqrt(un))
    if model['branch'] == 'sqrt_minus':
        return float(core - c * math.sqrt(un))
    return float(core - c * un)
