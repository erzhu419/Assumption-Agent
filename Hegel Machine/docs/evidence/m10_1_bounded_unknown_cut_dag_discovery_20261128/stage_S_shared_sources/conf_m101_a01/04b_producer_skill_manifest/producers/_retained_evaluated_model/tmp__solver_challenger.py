import numpy as np


def scalar_predict(x, q, y, poly):
    a, b = poly(x, 2), poly(q, 2)
    def kernel(u, v):
        r = (np.sqrt(5.) / 2.4) * np.linalg.norm(u[:, None, :] - v[None, :, :], axis=2)
        return (1. + r + r*r/3.) * np.exp(-r)
    k = kernel(x, x) + .001 * np.eye(len(x))
    w = np.linalg.solve(k, np.column_stack((a, y)))
    coef = np.linalg.solve(a.T @ w[:, :-1], a.T @ w[:, -1])
    return b @ coef + kernel(q, x) @ (w[:, -1] - w[:, :-1] @ coef)


def state_predict(study, poly, ridge, aff):
    a, b = study['panel_a'], study['panel_b']
    ua = aff(a['u'])
    keep = np.arange(len(ua))
    for _ in range(4):
        fit = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(ua @ fit - a['v'], axis=1))[:45]
    ca = np.linalg.lstsq(ua[keep], a['v'][keep], rcond=None)[0]
    cb = np.linalg.lstsq(aff(b['u']), b['v'], rcond=None)[0]
    def features(x):
        z = aff(aff(x) @ ca) @ cb
        d = 1. + .16*z*z
        return np.column_stack((poly(z, 2), z**3/d, z*z/d))
    tr = study['train']
    return ridge(features(tr['x']), features(study['test']['x']), tr['traj'], np.r_[0., [.01]*4, [1.]*18])
