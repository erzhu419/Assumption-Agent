import numpy as np
from scipy.optimize import least_squares

_model = None

def _matrix(rows, rates, terms):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    fast, slow = np.exp(rates)
    ef = np.exp(-h * fast)
    es = np.exp(-h * slow)
    sources = {'u': u, 's': s, 'one': np.ones_like(u)}
    cols = [np.ones_like(u), u, s]
    for source, horizon in terms:
        cols.append(sources[source] * (ef if horizon == 'f' else es))
    return np.column_stack(cols)

def fit(train_rows):
    global _model
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    candidates = []
    for a in ('u', 's', 'one'):
        for b in ('u', 's', 'one'):
            basic = [(a, 'f'), (b, 's')]
            candidates.append(basic)
            for extra in [('s', 'f'), ('s', 's')]:
                if extra not in basic:
                    candidates.append(basic + [extra])
    best = None
    for terms in candidates:
        def residual(z):
            X = _matrix(rows, z, terms)
            return X @ np.linalg.lstsq(X, y, rcond=None)[0] - y
        for start in ([0.0, -2.3], [1.0, -3.0], [-1.0, -2.0]):
            result = least_squares(residual, start, bounds=([-5, -5], [4, 4]), max_nfev=400)
            X = _matrix(rows, result.x, terms)
            coeff = np.linalg.lstsq(X, y, rcond=None)[0]
            err = X @ coeff - y
            score = float(err @ err) + 1e-7 * len(coeff)
            if best is None or score < best[0]:
                best = (score, result.x.copy(), terms, coeff)
    _model = best[1:]

def predict(row):
    rates, terms, coeff = _model
    return float(_matrix([row], rates, terms)[0] @ coeff)
