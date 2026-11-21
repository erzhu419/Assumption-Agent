import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    s = np.array([r.get('s_grad', 0.0) for r in rows], dtype=float)

    # Fast positive gain saturates quickly; slower adaptation subtracts
    # an approximately linear-in-dose term at long horizons.
    def pred(p, uu, hh, ss):
        af, tsf, aa, tss, rho = p
        fast = af * (1.0 - np.exp(-hh / tsf))
        slow = aa * (1.0 - np.exp(-hh / tss))
        sink = 1.0 - rho * ss / (1.0 + ss)
        return uu * (sink * fast - slow)

    def residual(p):
        return pred(p, u, h, s) - y

    # Positive amplitudes and ordered timescales avoid swapped/degenerate fits.
    best = None
    for x0 in ([1.5, 0.3, 0.7, 12.0, 0.0], [2.5, 0.5, 0.8, 20.0, 0.2], [1.0, 1.0, 0.5, 8.0, 0.5]):
        res = least_squares(residual, x0=x0, bounds=([0, 1e-4, 0, 0.5001, 0], [20, 5, 20, 1000, 1]), max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    af, tsf, aa, tss, rho = best.x
    return {'params': (float(af), float(tsf), float(aa), float(tss), float(rho))}

def predict(row):
    af, tsf, aa, tss, rho = fit._params
    u = float(row['u_now']); h = float(row['H']); s = float(row.get('s_grad', 0.0))
    return u * ((1.0 - rho*s/(1.0+s))*af*(1.0-math.exp(-h/tsf)) - aa*(1.0-math.exp(-h/tss)))

# Bind fitted parameters for predict after fit, as required by the harness.
_fit_impl = fit
def fit(train_rows):
    result = _fit_impl(train_rows)
    fit._params = result['params']
    return result
