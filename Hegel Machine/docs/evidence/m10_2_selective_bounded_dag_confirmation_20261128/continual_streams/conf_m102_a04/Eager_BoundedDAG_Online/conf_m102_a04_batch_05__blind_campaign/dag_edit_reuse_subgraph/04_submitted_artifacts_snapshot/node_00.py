import numpy as np
from scipy.optimize import least_squares


def _linear(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    p = u.shape[1]
    j = (np.arange(p) + 1) % p
    coef = []
    for i in range(p):
        a, b = u[:, i], u[:, j[i]]
        f = np.column_stack((np.ones(len(a)), a, b, a**3, a*b))
        y = v[:, i]
        c = least_squares(lambda c: f @ c - y, _linear(f, y),
                          jac=lambda c: f.copy(), loss='cauchy', f_scale=.04).x
        for _ in range(4):
            r = np.abs(f @ c - y)
            keep = r <= max(.001, 5*np.median(r))
            if keep.sum() >= f.shape[1]:
                c = _linear(f[keep], y[keep])
        coef.append(c)
    o, d, e, g, h = np.asarray(coef).T
    linear = np.diag(d)
    linear[np.arange(p), j] += e
    z0 = np.linalg.solve(linear, (observed-o).T).T
    def forward(z):
        w = z[j]
        return o + d*z + e*w + g*z**3 + h*z*w
    def jac(z):
        a = np.diag(d + 3*g*z*z + h*z[j])
        a[np.arange(p), j] += e + h*z
        return a
    return np.stack([least_squares(lambda z: forward(z)-y, z0i,
                                   jac=jac, gtol=1e-12).x
                     for y, z0i in zip(observed, z0)])


def _interactions(u):
    return np.column_stack((np.ones(len(u)), u*np.roll(u, -1, axis=1)))


def _curvature(u, q):
    w = u*u
    return np.column_stack((np.ones(len(u)), w, w/(1+q*w)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    n = len(train)
    z = _calibrate(study['panel_a'], np.vstack((train, study['test']['x'])))
    c = study['panel_c']
    u, v1, v2 = (np.asarray(c[k], dtype=float) for k in ('u', 'v1', 'v2'))
    first = _interactions(z) @ _linear(_interactions(u), v1)
    def residual(q):
        f = _curvature(u, q[0])
        return (f @ _linear(f, v2) - v2).ravel()
    q = least_squares(residual, [.25], bounds=(.02, 3.), gtol=1e-12).x[0]
    second = _curvature(z, q) @ _linear(_curvature(u, q), v2)
    features = np.column_stack((first, second))
    features = (features - features[:n].mean(0)) / np.maximum(features[:n].std(0), .1)
    return {'features': features, 'n_train': n}
