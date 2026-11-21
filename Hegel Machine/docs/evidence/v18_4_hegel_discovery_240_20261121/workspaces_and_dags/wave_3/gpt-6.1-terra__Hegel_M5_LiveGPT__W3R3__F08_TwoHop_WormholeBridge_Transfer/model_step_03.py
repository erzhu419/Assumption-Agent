import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(rows, q, kind, interaction):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    a, b = np.exp(q)
    if kind[0] == 'sat':
        fu = u / (a + u)
    else:
        fu = np.log1p(u / a)
    if kind[1] == 'sat':
        fs = s / (b + s)
    else:
        fs = np.log1p(s / b)
    cols = [np.ones(len(rows)), fu, fs]
    if interaction:
        cols.append(fu * fs)
    return np.column_stack(cols)

def _fit_candidate(rows, kind, interaction):
    y = np.array [r['y'] for r in rows]
    def residual(q):
        X = _features(rows, q, kind, interaction)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return X @ coef - y
    best = None
    for initial in ([0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]):
        result = least_squares(residual, initial, bounds=(-4.0, 4.0), max_nfev=500)
        loss = np.dot(result.fun, result.fun)
        if best is None or loss < best[0]:
            best = (loss, result.x)
    X = _features(rows, best[1], kind, interaction)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return best[1], coef

def fit(train_rows):
    global _model
    rows = list(train_rows)
    candidates = [(k, i) for k in [('sat','sat'), ('sat','log'), ('log','sat'), ('log','log')] for i in (False, True)]
    best = None
    for kind, interaction in candidates:
        errors = []
        for j in range(len(rows)):
            subset = rows[:j] + rows[j+1:]
            q, coef = _fit_candidate(subset, kind, interaction)
            pred = float(_features([rows[j]], q, kind, interaction)[0] @ coef)
            errors.append(pred - rows[j]['y'])
        score = float(np.mean(np.square(errors)))
        if best is None or score < best[0]:
            best = (score, kind, interaction)
    _, kind, interaction = best
    q, coef = _fit_candidate(rows, kind, interaction)
    _model = (kind, interaction, q, coef)

def predict(row):
    kind, interaction, q, coef = _model
    return float(_features([row], q, kind, interaction)[0] @ coef)
