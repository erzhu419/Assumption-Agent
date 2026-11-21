import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(row, p, variant):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    free_rho, sqrt_branch, multiplicative = variant
    i = 0
    vmax = p[i]; i += 1
    rho = p[i] if free_rho else 0.7
    i += int(free_rho)
    c = p[i] if sqrt_branch else 0.0
    i += int(sqrt_branch)
    af, aslow, log_tf, log_gap = p[i:i+4]
    tf = math.exp(log_tf)
    ts = tf + math.exp(log_gap)
    base = vmax * (1.0 - rho*s/(1.0+s)) * u/(1.2+u) + c*math.sqrt(max(u, 0.0))
    temporal = af*(-math.expm1(-h/tf)) - aslow*(-math.expm1(-h/ts))
    return base*(1.0+temporal) if multiplicative else base + u*temporal

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for free_rho in (False, True):
        for sqrt_branch in (False, True):
            for multiplicative in (False, True):
                variant = (free_rho, sqrt_branch, multiplicative)
                for tf, ts in ((0.5, 8.0), (2.0, 12.0), (5.0, 20.0), (0.1, 20.0)):
                    p0 = [3.0] + ([0.7] if free_rho else []) + ([0.0] if sqrt_branch else []) + [1.0, 2.0, math.log(tf), math.log(ts-tf)]
                    lo = [0.0] + ([0.0] if free_rho else []) + ([-10.0] if sqrt_branch else []) + [-20.0, -20.0, math.log(0.02), math.log(0.05)]
                    hi = [30.0] + ([2.0] if free_rho else []) + ([10.0] if sqrt_branch else []) + [20.0, 20.0, math.log(50.0), math.log(200.0)]
                    def residual(p):
                        return np.array([_value(r, p, variant) for r in rows]) - y
                    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                    rss = float(np.dot(result.fun, result.fun))
                    score = len(rows)*math.log(max(rss/len(rows), 1e-16)) + len(p0)*math.log(len(rows))
                    if best is None or score < best[0]:
                        best = (score, variant, result.x.copy())
    _model, _params = best[1], best[2]

def predict(row):
    return float(_value(row, _params, _model))