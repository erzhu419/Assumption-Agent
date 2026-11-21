import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] / 4.0 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    inhibition = lambda rho: 1.0 - rho * s / (1.0 + s)
    base = lambda vmax, rho: vmax * inhibition(rho) * u / (1.2 + u)
    candidates = []
    for branch in ('sqrt', 'leak'):
        def pred(p):
            vmax, rho, c = p
            term = c * np.sqrt(u) if branch == 'sqrt' else -c * u
            return base(vmax, rho) + term
        res = least_squares(lambda p: pred(p) - y, x0=[4.0, 0.5, 0.1], bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
        candidates.append((float(np.sum(res.fun ** 2)), branch, res.x))
    _, branch, p = min(candidates, key=lambda z: z[0])
    return {'branch': branch, 'params': p.tolist()}

def predict(row):
    state = fit(train_rows)
    vmax, rho, c = state['params']
    u = row['u_now'] / 4.0
    s = row['s_grad']
    value = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    if state['branch'] == 'sqrt':
        value += c * np.sqrt(u)
    else:
        value -= c * u
    return float(value)
