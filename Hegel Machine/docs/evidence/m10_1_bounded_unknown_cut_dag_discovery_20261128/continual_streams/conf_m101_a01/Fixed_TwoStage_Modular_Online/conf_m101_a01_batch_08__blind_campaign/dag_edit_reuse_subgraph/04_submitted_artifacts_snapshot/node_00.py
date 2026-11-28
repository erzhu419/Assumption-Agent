import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _panel_a(panel):
    design, target = _affine(panel['u']), panel['v']
    coef = _fit(design, target)
    for _ in range(20):
        residual = np.linalg.norm(design @ coef - target, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1.0 / np.hypot(1.0, residual / scale)
        coef = _fit(design * weight[:, None], target * weight[:, None])
    residual = np.linalg.norm(design @ coef - target, axis=1)
    keep = np.argsort(residual)[:-3]
    return _fit(design[keep], target[keep])


def _inverse(x, coef):
    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T


def _logcosh(x):
    return np.logaddexp(x, -x) - np.log(2.0)


def _readout(panel, key, nonlinearity, start, bounds):
    u, target = panel['u'], panel[key]

    def design(x, pars):
        frequency, gain = pars
        return _affine(x + gain * nonlinearity(frequency * x))

    def residual(pars):
        d = design(u, pars)
        return (d @ _fit(d, target) - target).ravel()

    pars = least_squares(residual, start, bounds=bounds).x
    coef = _fit(design(u, pars), target)
    return lambda x: design(x, pars) @ coef


def _softsign(x):
    return x / (1.0 + np.abs(x))


def _v2_curvature(panel):
    # Fit v2 = intercept + (u + curvature(u)) @ mixing.
    # Work in the unmixed coordinates for the scalar distance kernel.
    u, target = panel['u'], panel['v2']

    def residual(params):
        k = params[0]
        d = _affine(u + _logcosh(k * u) / k)
        return (d @ _fit(d, target) - target).ravel()

    k = least_squares(residual, (1.1,), bounds=((0.1,), (4.0,))).x[0]
    return lambda x: _logcosh(k * x) / k


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    x = np.concatenate((train['x'], test['x']), axis=0)
    a = _panel_a(study['panel_a'])
    b = _fit(_affine(study['panel_b']['u']), study['panel_b']['v'])
    latent = _inverse(x, a)
    ib = _affine(latent) @ b
    panel = study['panel_c']
    v1 = _readout(panel, 'v1', _softsign, (0.83, -7.3), ((0.05, -30), (15, 0)))
    curvature = _v2_curvature(panel)
    r = _inverse(latent, b)
    i2b = _inverse(r - 0.25 * curvature(r), b)
    return {'z': {'x': x, 'ib': ib, 'i2b': i2b, 'i1': v1(ib)},
            'y': train['y'], 'traj': train['traj']}
