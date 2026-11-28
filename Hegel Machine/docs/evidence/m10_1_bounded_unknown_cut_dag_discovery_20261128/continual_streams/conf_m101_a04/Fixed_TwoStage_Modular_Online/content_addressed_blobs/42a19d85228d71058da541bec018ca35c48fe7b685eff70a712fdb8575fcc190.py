import numpy as np
from scipy.optimize import least_squares


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _cyclic(u):
    return np.stack((np.ones_like(u), u, np.roll(u, -1, axis=1) * (u - 1), u**3), axis=-1)


def _calibrate(panel, observed):
    u, v = panel['u'], panel['v']
    design = _cyclic(u)
    w = np.ones(len(u))
    for _ in range(25):
        q = np.sqrt(w)
        g = np.array([np.linalg.lstsq(design[:, j] * q[:, None], v[:, j] * q, rcond=None)[0]
                      for j in range(u.shape[1])])
        e = np.linalg.norm(np.einsum('njk,jk->nj', design, g) - v, axis=1)
        w = 1 / (1 + (e / max(2 * np.median(e), 1e-8))**2)

    offset, linear, coupling, cubic = g.T
    idx = np.arange(u.shape[1])
    nxt = np.roll(idx, -1)

    def forward(z):
        return offset + linear * z + coupling * z[nxt] * (z - 1) + cubic * z**3

    def jac(z):
        j = np.diag(linear + coupling * z[nxt] + 3 * cubic * z*z)
        j[idx, nxt] += coupling * (z - 1)
        return j

    j0 = jac(np.zeros(u.shape[1]))
    initial = np.linalg.solve(j0, (observed - offset).T).T
    return np.array([least_squares(lambda z: forward(z) - row, z0, jac=jac, gtol=1e-12).x
                     for row, z0 in zip(observed, initial)])


def _c1(u, a):
    return u + a[1] * np.arcsinh(a[0] * u) + a[2] * u**3


def _c2(u, a):
    return u + a[1] * (np.cos(a[0] * u) - 1) + a[2] * u**2


def _project(u, v, z, transform, initial):
    def fit(a):
        design = _affine(transform(u, a))
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        return design @ weights - v, weights

    a = least_squares(lambda a: fit(a)[0].ravel(), initial, max_nfev=180).x
    return _affine(transform(z, a)) @ fit(a)[1]


def _branches(study, observed):
    z = _calibrate(study['panel_a'], observed)
    b, c = study['panel_b'], study['panel_c']
    b_fit = np.linalg.lstsq(_affine(b['u']), b['v'], rcond=None)[0]
    v1 = _project(c['u'], c['v1'], z, _c1, [4.3, 4.4, -.56])
    v2 = _project(c['u'], c['v2'], z, _c2, [2.5, -.06, .35])
    return z, _affine(z) @ b_fit, v1, v2


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    z, b, v1, v2 = _branches(study, np.concatenate((train['x'], test['x'])))
    return {'h': np.column_stack((b, v1, v2)), 'n_train': np.array(len(train['x'])),
            'y': np.asarray(train['y']), 'traj': np.asarray(train['traj'])}
