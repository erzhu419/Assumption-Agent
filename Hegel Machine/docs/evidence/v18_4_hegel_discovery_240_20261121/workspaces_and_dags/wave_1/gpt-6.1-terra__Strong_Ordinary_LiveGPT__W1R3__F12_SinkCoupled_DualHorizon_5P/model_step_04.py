import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    sink = s / (1.0 + s)
    base = u / (1.2 + u)
    best = None

    # Compare additive and sink-coupled temporal responses, with and without
    # a local square-root branch. All coefficients and time scales are fitted.
    for branch in (False, True):
        for coupling in ('u', 'base', 'sink_u', 'sink_base'):
            for shape in ('dual', 'delayed'):
                def design(logt, uu=u, ss=sink, bb=base, hh=h):
                    t1, t2 = np.exp(logt)
                    cols = [bb, -bb * ss]
                    if branch:
                        cols.append(np.sqrt(np.maximum(uu, 0.0)))
                    factor = {'u': uu, 'base': bb, 'sink_u': uu * (1.0 - 0.7 * ss), 'sink_base': bb * (1.0 - 0.7 * ss)}[coupling]
                    fast = -np.expm1(-hh / t1)
                    if shape == 'dual':
                        slow = -np.expm1(-hh / t2)
                    else:
                        slow = -np.expm1(-(hh / t2) ** 2)
                    cols.extend([factor * fast, -factor * slow])
                    return np.column_stack(cols)

                def residual(logt):
                    X = design(logt)
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    return X @ coef - y

                for start in ((0.3, 10.0), (1.0, 15.0), (3.0, 20.0), (5.0, 8.0)):
                    result = least_squares(residual, np.log(start), bounds=(np.log([0.03, 0.5]), np.log([50.0, 150.0])), max_nfev=1500)
                    X = design(result.x)
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    rss = float(np.sum((X @ coef - y) ** 2))
                    k = len(coef) + 2
                    score = len(y) * math.log(max(rss / len(y), 1e-14)) + k * math.log(len(y))
                    if best is None or score < best[0]:
                        best = (score, branch, coupling, shape, np.exp(result.x), coef)
    _model = best

def predict(row):
    _, branch, coupling, shape, times, coef = _model
    u = float(row['u_now'])
    s = float(row['s_grad']) / (1.0 + float(row['s_grad']))
    h = float(row['H'])
    b = u / (1.2 + u)
    factor = {'u': u, 'base': b, 'sink_u': u * (1.0 - 0.7 * s), 'sink_base': b * (1.0 - 0.7 * s)}[coupling]
    terms = [b, -b * s]
    if branch:
        terms.append(math.sqrt(max(u, 0.0)))
    terms.append(factor * (-math.expm1(-h / times[0])))
    z = h / times[1]
    terms.append(-factor * (-math.expm1(-z if shape == 'dual' else -z * z)))
    return float(np.dot(coef, terms))