import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    # A two-branch saturating response; each branch may depend on both
    # upstream supply and the downstream pull gradient.
    def calc(p, u, s):
        a, b, c, d, e, f = p
        return a + b / (u + np.exp(c)) + d * (s - 1.0) / (u + np.exp(e) + np.exp(f) * s)

    starts = [
        [3.2, -3.3, 0.0, -2.0, 0.0, 0.0],
        [3.2, -3.3, 0.0, -5.0, -1.0, 0.0],
        [3.2, -3.3, 0.0, -2.0, 1.0, -1.0],
    ]
    best = min((least_squares(lambda p: calc(p, u, s) - y, p0, max_nfev=5000) for p0 in starts), key=lambda z: np.sum(z.fun ** 2))

    # A rational surface provides an alternative if the separable two-branch
    # form is not supported by the observations.
    def rational(p, u, s):
        a, b, c, d, e, f, g = p
        den = u + np.exp(e) + (s - 1.0) * (f + g * u)
        den = np.where(np.abs(den) < 1e-8, np.where(den < 0, -1e-8, 1e-8), den)
        return (a * u + b + (s - 1.0) * (c * u + d)) / den

    q0 = [3.2, -0.2, 0.0, 0.0, 0.0, 0.2, 0.1]
    q = least_squares(lambda p: rational(p, u, s) - y, q0, max_nfev=5000)
    if np.mean(q.fun ** 2) < 0.5 * np.mean(best.fun ** 2):
        _model = ('rational', q.x)
    else:
        _model = ('branches', best.x)

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    kind, p = _model
    if kind == 'branches':
        a, b, c, d, e, f = p
        return float(a + b / (u + np.exp(c)) + d * (s - 1.0) / (u + np.exp(e) + np.exp(f) * s))
    a, b, c, d, e, f, g = p
    den = u + np.exp(e) + (s - 1.0) * (f + g * u)
    if abs(den) < 1e-8:
        den = -1e-8 if den < 0 else 1e-8
    return float((a * u + b + (s - 1.0) * (c * u + d)) / den)
