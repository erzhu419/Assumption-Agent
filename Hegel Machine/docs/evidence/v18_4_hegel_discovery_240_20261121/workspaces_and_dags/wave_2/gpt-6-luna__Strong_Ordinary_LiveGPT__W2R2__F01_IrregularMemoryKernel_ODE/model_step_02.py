import math
import numpy as np
from scipy.optimize import least_squares

def _history_z(row):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    pts = sorted(pts, key=lambda p: p['t'])
    z = 0.52 * float(pts[0]['x'])
    t0 = float(pts[0]['t'])
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dt = float(p1['t']) - float(p0['t'])
        if dt <= 0:
            continue
        x0, x1 = float(p0['x']), float(p1['x'])
        slope = (x1 - x0) / dt
        e = math.exp(-dt / _history_z.tau)
        # Exact integral for linearly interpolated x over this interval.
        z = e*z + 0.52*(x0*(1.0-e) + slope*(dt - _history_z.tau*(1.0-e)))
    return z

def _predict_params(row, p):
    a, b, c, tau = [float(v) for v in p]
    _history_z.tau = tau
    x = float(row['x_now'])
    z = _history_z(row)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    def residual(p):
        return np.asarray([_predict_params(r,p)-float(r['y']) for r in train_rows])
    best = None
    for start in ([0.3,0.3,0.5,0.3],[0.8,0.2,1.0,0.1],[0.1,0.8,0.2,0.8]):
        result = least_squares(residual, start, bounds=([0.001,0.001,-5.0,0.01],[5.0,5.0,5.0,5.0]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return {'params': best.x.tolist()}

def predict(row):
    return float(_predict_params(row, predict.params))

predict.params = [0.3,0.3,0.5,0.3]
# fit returns fitted parameters; callers should assign them to predict.params before prediction.
