import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.05

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('fit requires training rows')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    un = np.asarray([float(r['u_now']) / _TAU for r in rows], dtype=float)
    sg = np.asarray([float(r['s_grad']) for r in rows], dtype=float)
    base = (un / (1.2 + un))
    inhibition = sg / (1.0 + sg)
    best = None
    # Compare the documented signed square-root branch and linear-leak branch.
    for kind in ('sqrt_pos', 'sqrt_neg', 'leak'):
        branch = np.sqrt(np.maximum(un, 0.0)) if kind != 'leak' else un
        def pred(p):
            vmax, rho, amp = p
            core = vmax * (1.0 - rho * inhibition) * base
            return core + (amp * branch if kind == 'sqrt_pos' else (-amp * branch if kind == 'sqrt_neg' else -amp * branch))
        def residual(p):
            return pred(p) - y
        starts = ([2.0, 0.7, 0.1], [1.0, 0.3, 0.0], [4.0, 0.9, 0.5])
        for x0 in starts:
            res = least_squares(residual, x0, bounds=([-100.0, -10.0, -100.0], [100.0, 10.0, 100.0]), max_nfev=5000)
            score = float(np.mean(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, kind, res.x)
    _PARAMS = (best[1], np.asarray(best[2], dtype=float))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('call fit(train_rows) before predict(row)')
    kind, p = _PARAMS
    un = float(row['u_now']) / _TAU
    sg = float(row['s_grad'])
    base = un / (1.2 + un)
    inhibition = sg / (1.0 + sg)
    branch = math.sqrt(max(0.0, un)) if kind != 'leak' else un
    vmax, rho, amp = p
    core = vmax * (1.0 - rho * inhibition) * base
    return float(core + (amp * branch if kind == 'sqrt_pos' else -amp * branch))
