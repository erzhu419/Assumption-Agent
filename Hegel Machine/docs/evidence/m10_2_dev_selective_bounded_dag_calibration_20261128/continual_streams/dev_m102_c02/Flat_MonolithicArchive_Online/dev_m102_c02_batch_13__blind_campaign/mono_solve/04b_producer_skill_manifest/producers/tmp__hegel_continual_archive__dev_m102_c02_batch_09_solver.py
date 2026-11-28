"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _fit(h, y):
    return np.linalg.lstsq(h, y, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, rows):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    norms = np.linalg.norm(v, axis=1)
    keep = norms < max(4., 3*np.median(norms))
    eye = np.eye(u.shape[1])

    def transfers(x, p):
        a, e, b, f = p
        r = np.roll(x, 2, axis=-1)
        g = x / np.sqrt(1+a*np.abs(x)**e)
        h = r / (1+b*np.abs(r)**f)
        return g, h

    def design(x, p):
        g, h = transfers(x, p)
        return np.concatenate((np.broadcast_to(eye, (len(x), *eye.shape)),
                               g[:, :, None]*eye, h[:, :, None]*eye), axis=2)

    ncol = 3*u.shape[1]

    def residual(p):
        h = design(u[keep], p).reshape(-1, ncol)
        y = v[keep].ravel()
        return h @ _fit(h, y) - y

    p = least_squares(residual, [.22, 2.15, .8, 1.2],
                      bounds=([.001,.3,.001,.3], [8,5,8,5]), max_nfev=140).x
    h = design(u[keep], p).reshape(-1, ncol)
    offset, gain, mix = _fit(h, v[keep].ravel()).reshape(3, -1)

    def forward(x):
        g, h = transfers(x, p)
        return offset + gain*g + mix*h

    return np.array([least_squares(lambda z: forward(z)-x, x, gtol=1e-11).x
                     for x in rows])


def _ridge(x, y, n, penalty):
    mu = x[:n].mean(axis=0)
    h = (x-mu)/np.maximum(x[:n].std(axis=0), 1e-8)
    ht = h[:n]
    center = y.mean(axis=0)
    coef = np.linalg.solve(ht.T@ht + penalty*np.eye(h.shape[1]), ht.T@(y-center))
    return h[n:]@coef + center


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    z = _calibrate(study['panel_a'], np.concatenate((tr['x'], study['test']['x'])))
    panel = study['panel_b']
    b = _aff(z) @ _fit(_aff(panel['u']), panel['v'])
    panel = study['panel_c']
    u, v = panel['u'], panel['v2']

    def soft(x, p):
        return _aff(np.logaddexp(0, p[0]*x+p[1]))

    def panel_error(p):
        h = soft(u,p)
        return (h@_fit(h,v)-v).ravel()

    p = least_squares(panel_error, [2.2,0], gtol=1e-12).x
    d = soft(z,p) @ _fit(soft(u,p),v)
    pair = b[:,0]*b[:,2] + b[:,1]*b[:,3]
    y = _ridge(np.column_stack((b,pair)), np.asarray(tr['y']), n, .0002)
    state = _ridge(np.column_stack((b,b/(1+.22*d*d))), np.asarray(tr['traj']), n, .01)
    end = state[:,-1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(),y.std(),end.mean(),end.std()])}
