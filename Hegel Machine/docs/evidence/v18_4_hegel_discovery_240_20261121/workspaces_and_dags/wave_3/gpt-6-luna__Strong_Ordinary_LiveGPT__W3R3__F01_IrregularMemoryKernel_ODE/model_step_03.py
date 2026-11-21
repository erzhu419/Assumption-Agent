import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    pref = row.get('raw_prefix') or []
    if not pref:
        return float(row.get('mem_conv') if row.get('mem_conv') is not None else 0.52 * float(row['x_now']))
    tau = _TAU[0]
    z = 0.52 * float(pref[0]['x'])
    tprev = float(pref[0]['t'])
    xprev = float(pref[0]['x'])
    tlast = float(pref[-1]['t'])
    for item in pref[1:]:
        t = float(item['t']); x = float(item['x'])
        dt = t - tprev
        e = math.exp(-dt / tau)
        slope = (x - xprev) / dt if dt else 0.0
        z = e*z + 0.52*(xprev*(1.0-e) + slope*(dt - tau*(1.0-e)))
        tprev, xprev = t, x
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    pref = row.get('raw_prefix') or []
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    elif pref:
        # Reconstruct using the fitted fast time constant.
        z = 0.52 * float(pref[0]['x'])
        tprev = float(pref[0]['t']); xprev = float(pref[0]['x'])
        for item in pref[1:]:
            t = float(item['t']); x = float(item['x']); dt = t-tprev
            e = math.exp(-dt/tau)
            slope = (x-xprev)/dt if dt else 0.0
            z = e*z + 0.52*(xprev*(1-e) + slope*(dt-tau*(1-e)))
            tprev, xprev = t, x
    else:
        z = 0.52 * float(row['x_now'])
    x = float(row['x_now']); u = float(row['u_now']); H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004))); dt = H/n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx-zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2); k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _TAU
    _TAU = [1.0]
    rows = [r for r in train_rows if 'y' in r]
    if len(rows) < 4:
        raise ValueError('Training rows must include target y values.')
    def residual(p):
        _TAU[0] = max(float(p[3]), 1e-6)
        return np.array([_simulate(r,p)-float(r['y']) for r in rows])
    result = least_squares(residual, [0.6, 0.3, 0.4, 0.15], bounds=([0, -5, -5, 0.005], [5, 5, 5, 5]), max_nfev=3000)
    _TAU[0] = float(result.x[3])
    return result.x

def predict(row):
    return float(_simulate(row, _PARAMS))

_PARAMS = None
_OLD_FIT = fit
def fit(train_rows):
    global _PARAMS
    _PARAMS = _OLD_FIT(train_rows)
    return _PARAMS
