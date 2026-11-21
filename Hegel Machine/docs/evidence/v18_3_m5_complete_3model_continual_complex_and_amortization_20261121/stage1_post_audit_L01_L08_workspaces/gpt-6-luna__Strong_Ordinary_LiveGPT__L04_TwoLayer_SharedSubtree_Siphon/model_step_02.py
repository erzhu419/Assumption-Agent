import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    # Fit the shared sink-modulated saturating pool and compare the three
    # documented branch forms, including both possible sqrt signs.
    def base(p, row):
        vmax, rho = p[:2]
        u = row['u_now']
        s = row['s_grad']
        un = u / 1.0  # supplied characteristic tau_scale
        return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))

    candidates = []
    for branch in ('sqrt+', 'sqrt-', 'leak'):
        def pred(p, row):
            b = base(p, row)
            u = row['u_now']
            if branch == 'sqrt+':
                return b + p[2] * math.sqrt(max(u, 0.0))
            if branch == 'sqrt-':
                return b - p[2] * math.sqrt(max(u, 0.0))
            return b - p[2] * u
        def residual(p):
            return np.array([pred(p, r) - r['y'] for r in rows])
        result = least_squares(residual, x0=[5.5, 0.7, 0.05], bounds=([0.01, 0.0, 0.0], [100.0, 1.0, 100.0]))
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, branch, p = min(candidates, key=lambda item: item[0])
    return {'branch': branch, 'vmax': float(p[0]), 'rho': float(p[1]), 'branch_scale': float(p[2])}

def predict(row):
    p = fit(_FIT_ROWS)
    u = row['u_now']
    value = p['vmax'] * (1.0 - p['rho'] * row['s_grad'] / (1.0 + row['s_grad'])) * (u / (1.2 + u))
    if p['branch'] == 'sqrt+':
        value += p['branch_scale'] * math.sqrt(max(u, 0.0))
    elif p['branch'] == 'sqrt-':
        value -= p['branch_scale'] * math.sqrt(max(u, 0.0))
    else:
        value -= p['branch_scale'] * u
    return float(value)
