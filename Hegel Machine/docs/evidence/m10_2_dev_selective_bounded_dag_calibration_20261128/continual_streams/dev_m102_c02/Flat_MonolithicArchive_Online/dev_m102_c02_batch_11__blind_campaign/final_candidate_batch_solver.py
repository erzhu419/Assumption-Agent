"""Calibrate coordinates, apply the panel B operator, and predict both readouts."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    norms = np.linalg.norm(v, axis=1)
    keep = norms <= max(4., 3.*np.median(norms))
    u, v = u[keep], v[keep]
    other = np.roll(u, 2, axis=1)

    def warp(z, p):
        return z + p[0]*np.arctan(p[1]*z)

    def profile(p):
        w = warp(u, p)
        sin = np.sin(p[2]*other)
        w0 = w-w.mean(axis=0)
        sin0 = sin-sin.mean(axis=0)
        v0 = v-v.mean(axis=0)
        ww = (w0*w0).sum(axis=0)
        ws = (w0*sin0).sum(axis=0)/ww
        wv = (w0*v0).sum(axis=0)/ww
        sin_res = sin0-w0*ws
        cross = np.sum(sin_res*v0)/np.sum(sin_res**2)
        gain = wv-cross*ws
        bias = (v-gain*w-cross*sin).mean(axis=0)
        return bias+gain*w+cross*sin-v, (bias, gain, cross)

    p = least_squares(lambda p: profile(p)[0].ravel(), [.26,1.75,2.4],
                      bounds=([.01,.2,1.2],[2.,5.,3.6]), gtol=1e-11).x
    bias, gain, cross = profile(p)[1]
    target = (x-bias)/gain
    z = target/(1+p[0]*p[1])
    # Simultaneous Newton updates of the two independent coordinate pairs.
    for _ in range(16):
        q = np.roll(z, 2, axis=1)
        r = bias+gain*warp(z, p)+cross*np.sin(p[2]*q)-x
        a = gain*(1+p[0]*p[1]/(1+(p[1]*z)**2))
        b = cross*p[2]*np.cos(p[2]*q)
        step = (np.roll(a,2,axis=1)*r-b*np.roll(r,2,axis=1))/(a*np.roll(a,2,axis=1)-b*np.roll(b,2,axis=1))
        z -= np.clip(step, -.5, .5)
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return z


def _panel_map(panel, key, fn, start, bounds, z):
    u, v = panel['u'], panel[key]
    def residual(p):
        h = _aff(fn(u, p))
        return (h @ _fit(h, v) - v).ravel()
    p = least_squares(residual, start, bounds=bounds, gtol=1e-12).x
    return _aff(fn(z, p)) @ _fit(_aff(fn(u, p)), v)


def _maps(study, x):
    z = _calibrate(study['panel_a'], x)
    pb, pc = study['panel_b'], study['panel_c']
    b = _aff(z) @ _fit(_aff(pb['u']), pb['v'])
    az = np.concatenate((z, b))
    c1, b1 = np.split(_panel_map(pc, 'v1', lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2], [.62,1.07,1.57], ([0,.1,.1],[5,5,5]), az), 2)
    c2, b2 = np.split(_panel_map(pc, 'v2', lambda x,p:np.logaddexp(0,p[0]*x+p[1]), [2.2,0], ([.1,-3],[8,3]), az), 2)
    c = _panel_map(pc, 'v1', lambda x,p:np.logaddexp(0,p[0]*x+p[1]), [2.15,0], ([.1,-4],[10,4]), z)
    d = _panel_map(pc, 'v2', lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1], [.22,1.25], ([0,.25],[.8,4]), z)
    return z,b,c1,c2,b1,b2,c,d


def _designs(study, x):
    z,b,c1,c2,b1,b2,c,d = _maps(study, x)
    a = float(study['telemetry_vec'][0])
    pair = np.sum(b[:,:2]*b[:,2:],axis=1)
    h0 = np.column_stack((_aff(b), pair, c, d, c/(1+a*z*z)))
    h1 = np.column_stack((h0, np.sum(b*d,axis=1)))
    h2 = np.column_stack((h0, c/(1+a*b*b)))
    h3 = np.column_stack((_aff(z), c1*b1, c1*c2, c1/(1+a*z*z)))
    h4 = np.column_stack((_aff(z), c1*b1, c1*b, c1/(1+a*z*z)))
    return _aff(b), (h1,h2,h3,h4)


def _scalar_member(h, y, n, penalty, guard):
    h = h[:, 1:]
    mu, sd = h[:n].mean(0), np.maximum(h[:n].std(0), 1e-9)
    a, q = np.split((h-mu)/sd, [n])
    u, d, v = np.linalg.svd(a, full_matrices=False)
    w = q @ v.T
    center = y.mean()
    r = u.T @ (y-center)
    base = center + w @ (r*d/(d*d+penalty))
    if guard:
        stable = center + w @ (r*d/(d*d+max(.10, penalty)))
        leverage = (w*w/(d*d+penalty)).sum(axis=1)
        mix = np.clip((leverage-guard)/np.maximum(leverage, 1e-12), 0., 1.)
        base += mix*(stable-base)
    return base

def solve_batch(study: dict) -> dict:
    """Fit panel-calibrated scalar readouts and an affine trajectory map."""
    tr, te = study['train'], study['test']
    n = len(tr['x'])
    h, designs = _designs(study, np.concatenate((tr['x'], te['x'])))
    members = [_scalar_member(q, tr['y'], n, p, 0.0)
               for q, p in zip(designs, (0.001, 0.001, 0.01, 0.01))]
    y = (0.35/2)*(members[0]+members[1]) + ((1-0.35)/2)*(members[2]+members[3])
    state = h[n:] @ _fit(h[:n], tr['traj'])
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
