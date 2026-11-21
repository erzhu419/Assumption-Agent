import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    tau = 4.0
    u = np.array([max(0.0, r['u_now'] / tau) for r in rows], dtype=float)
    s = np.array([max(0.0, r['s_grad']) for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    rootu = np.sqrt(u)
    sat = u / (1.2 + u)
    inhib = s / (1.0 + s)
    candidates = []
    # Fit the disclosed two-layer siphon law, checking the plausible open-branch signs/forms.
    for branch in ('plus_sqrt', 'minus_sqrt', 'linear_leak'):
        def pred(p):
            vmax, rho, c = p
            base = vmax * (1.0 - rho * inhib) * sat
            if branch == 'plus_sqrt':
                return base + c * rootu
            if branch == 'minus_sqrt':
                return base - c * rootu
            return base - c * u
        res = least_squares(lambda p: pred(p) - y,
                            x0=np.array([2.5, 0.5, 0.2]),
                            bounds=([0.0, 0.0, -10.0], [20.0, 1.0, 10.0]),
                            max_nfev=5000)
        candidates.append((float(np.mean((pred(res.x) - y) ** 2)), branch, res.x))
    _, branch, pars = min(candidates, key=lambda z: z[0])
    return {'branch': branch, 'params': [float(v) for v in pars]}

def predict(row):
    model = fit([]) if False else None
    # Parameters are fitted on train_rows; cache is attached by fit for prediction.
    return float(_predict_with_model(row, _MODEL))

_MODEL = None

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    tau = 4.0
    u = np.array([max(0.0, r['u_now'] / tau) for r in rows], dtype=float)
    s = np.array([max(0.0, r['s_grad']) for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    rootu = np.sqrt(u); sat = u / (1.2 + u); inhib = s / (1.0 + s)
    opts = []
    for branch in ('plus_sqrt', 'minus_sqrt', 'linear_leak'):
        def calc(p):
            base = p[0] * (1.0 - p[1] * inhib) * sat
            return base + p[2] * rootu if branch == 'plus_sqrt' else (base - p[2] * rootu if branch == 'minus_sqrt' else base - p[2] * u)
        res = least_squares(lambda p: calc(p) - y, [2.5, 0.5, 0.2], bounds=([0,0,-10],[20,1,10]), max_nfev=5000)
        opts.append((float(np.mean((calc(res.x)-y)**2),), branch, res.x))
    _, branch, p = min(opts, key=lambda q:q[0])
    _MODEL = (branch, [float(v) for v in p])
    return _MODEL

def _predict_with_model(row, model):
    branch, p = model
    u = max(0.0, row['u_now'] / 4.0); s = max(0.0, row['s_grad'])
    base = p[0] * (1.0 - p[1] * s/(1.0+s)) * u/(1.2+u)
    return base + p[2]*math.sqrt(u) if branch == 'plus_sqrt' else (base - p[2]*math.sqrt(u) if branch == 'minus_sqrt' else base - p[2]*u)
