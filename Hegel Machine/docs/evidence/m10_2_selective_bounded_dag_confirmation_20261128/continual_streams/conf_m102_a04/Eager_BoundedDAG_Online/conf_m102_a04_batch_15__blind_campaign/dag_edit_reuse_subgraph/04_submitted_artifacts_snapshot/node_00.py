import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _align(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    f = one(u)
    pilot = least_squares(lambda q: (f @ q.reshape(f.shape[1], -1) - v).ravel(), lin(f, v).ravel(), loss='cauchy', f_scale=.18).x.reshape(f.shape[1], -1)
    keep = np.max(abs(f @ pilot - v), axis=1) < 2.
    j = np.arange(u.shape[1]); k = (j+1) % len(j)
    def model(w, p):
        return p[0] + p[1]*w + p[2]*w[:, k] + p[3]*w*w[:, k] + p[4]*w**3
    p = np.column_stack([lin(np.c_[np.ones(sum(keep)), u[keep, a], u[keep, b], u[keep, a]*u[keep, b], u[keep, a]**3], v[keep, a]) for a,b in zip(j,k)])
    def jac(z):
        w = z.reshape(x.shape)
        g = np.zeros((len(w), len(j), len(j)))
        g[:,j,j] = p[1]+p[3]*w[:,k]+3*p[4]*w*w
        g[:,j,k] = p[2]+p[3]*w
        return g
    w = (x-p[0])/p[1]
    for _ in range(16):
        step = np.linalg.solve(jac(w), (model(w,p)-x)[...,None])[...,0]
        w -= np.clip(step, -.4, .4)
        if np.max(abs(step)) < 1e-11:
            break
    return w


def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        rate, amplitude, interaction = q
        return one(w + amplitude*np.tanh(rate*w) + interaction*w*np.roll(w, 1, axis=1))

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [1.2, .27, .146], ([.2, .001, -.7], [4., 2., .7]), gtol=1e-12)
    return np.c_[c, d]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x, test = study['train']['x'], study['test']['x']
    n = len(x)
    z = _align(study['panel_a'], np.vstack((x, test)))
    w = _panel_maps(study['panel_c'], z)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    return {'features': w, 'n_train': n}
