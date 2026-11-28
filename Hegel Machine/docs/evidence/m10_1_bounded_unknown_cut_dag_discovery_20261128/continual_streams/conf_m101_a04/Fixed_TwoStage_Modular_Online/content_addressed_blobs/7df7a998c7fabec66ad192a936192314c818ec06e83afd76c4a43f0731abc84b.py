import numpy as np
from scipy.optimize import minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _ring(u):
    return u * np.roll(u, -1, axis=1)


def _calibration(u):
    return _design(u, _ring(u), u**3)


def _decode(panel, observed):
    u, v = panel['u'], panel['v']
    d = _calibration(u)
    w = np.ones(len(u))
    for _ in range(20):
        c = _fit(d * w[:, None], v * w[:, None])
        r = np.linalg.norm(v - d @ c, axis=1)
        w = 1 / np.hypot(1, r / max(2 * np.median(r), 1e-10))
    keep = np.argsort(r)[:-3]
    c = _fit(d[keep], v[keep])
    z = np.linalg.solve(c[1:5].T, (observed - c[0]).T).T
    i = np.arange(z.shape[1])
    j = (i + 1) % z.shape[1]
    for _ in range(16):
        jac = np.broadcast_to(c[1:5], (len(z), 4, 4)).copy()
        jac += 3 * z[:, :, None]**2 * c[9:13]
        jac[:, i, :] += z[:, j, None] * c[5:9]
        jac[:, j, :] += z[:, i, None] * c[5:9]
        step = np.linalg.solve(jac.transpose(0, 2, 1), (_calibration(z) @ c - observed)[..., None])[..., 0]
        z -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return z


def _v2(panel, z):
    u, v = panel['u'], panel['v2']

    def basis(x, shape):
        even = x*x / (np.sqrt(1 + shape*x*x) + 1)
        return _design(x, even)

    def loss(shape):
        d = basis(u, shape)
        return np.sum((d @ _fit(d, v) - v)**2)

    shape = minimize_scalar(loss, bounds=(0.01, 3.0), method='bounded').x
    return basis(z, shape) @ _fit(basis(u, shape), v)


def _features(study):
    train, test = study['train'], study['test']
    z = _decode(study['panel_a'], np.concatenate((train['x'], test['x'])))
    b, c = study['panel_b'], study['panel_c']
    b_hat = _design(z) @ _fit(_design(b['u']), b['v'])
    c1_hat = _design(_ring(z)) @ _fit(_design(_ring(c['u'])), c['v1'])
    c2_hat = _v2(c, z)
    return b_hat, c1_hat, c2_hat


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    b, c1, c2 = _features(study)
    n_train = len(train['x'])
    # Scale the interactions for the frozen ridge readout while retaining B's
    # affine information. Center on training rows only.
    h = np.column_stack((0.09 * b, 2.0 * c1, c2))
    h -= h[:n_train].mean(axis=0)
    return {'h': h,
            'n_train': np.asarray(n_train),
            'y': train['y'], 'traj': train['traj']}
