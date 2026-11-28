import numpy as np


def _basis(x):
    """For each channel, use its own input and its cyclic successor."""
    z = np.roll(x, -1, axis=1)
    return np.stack((np.ones_like(x), x, z, x*z, x**3), axis=-1)


def _calibrate(u, v):
    a = _basis(u)

    def fit(rows, w):
        return np.stack([
            np.linalg.lstsq(a[rows, j]*w[:, None], v[rows, j]*w, rcond=None)[0]
            for j in range(u.shape[1])
        ])

    rows = np.arange(len(u))
    w = np.ones(len(u))
    # Shared Cauchy weights screen contamination at the row level.
    for _ in range(20):
        c = fit(rows, w)
        r = np.linalg.norm(np.sum(a*c, axis=-1)-v, axis=1)
        scale = max(2*np.median(r), 1e-9)
        w = scale/np.hypot(scale, r)
    keep = np.argsort(r)[:-3]
    return fit(keep, np.ones(len(keep)))


def _forward(x, c):
    return np.sum(_basis(x)*c, axis=-1)


def _jacobian(x, c):
    j = np.arange(x.shape[1])
    k = (j+1) % x.shape[1]
    jac = np.zeros((len(x), len(j), len(j)))
    jac[:, j, j] = c[:, 1] + c[:, 3]*x[:, k] + 3*c[:, 4]*x**2
    jac[:, j, k] = c[:, 2] + c[:, 3]*x
    return jac


def _invert(y, c):
    linear = _jacobian(np.zeros_like(y[:1]), c)[0]
    x = np.linalg.solve(linear, (y-c[:, 0]).T).T
    for _ in range(30):
        r = _forward(x, c)-y
        if np.max(np.abs(r)) < 1e-11:
            break
        step = np.linalg.solve(_jacobian(x, c), r[:, :, None])[:, :, 0]
        trial = x-step
        norm = np.linalg.norm(r, axis=1)
        for _ in range(12):
            bad = np.linalg.norm(_forward(trial, c)-y, axis=1) > norm
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = x[bad]-step[bad]
        x = trial
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, c = study['panel_a'], study['panel_c']
    coef = _calibrate(a['u'], a['v'])
    measured = np.concatenate((study['train']['x'], study['test']['x']))
    return {'u': c['u'], 'v1': c['v1'], 'v2': c['v2'], 'x': _invert(measured, coef)}
