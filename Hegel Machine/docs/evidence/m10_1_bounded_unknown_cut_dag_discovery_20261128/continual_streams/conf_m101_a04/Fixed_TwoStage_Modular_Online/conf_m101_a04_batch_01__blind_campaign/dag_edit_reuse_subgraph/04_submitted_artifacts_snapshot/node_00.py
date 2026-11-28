import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibration(panel):
    design = _affine(panel['u'])
    values = panel['v']
    weights = np.ones(len(design))
    for _ in range(20):
        coef = _fit(design * weights[:, None], values * weights[:, None])
        residual = np.linalg.norm(values - design @ coef, axis=1)
        scale = max(np.median(residual), 1e-10)
        weights = 1.0 / np.hypot(1.0, residual / scale)
    return coef


def _bilinear(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _logcosh_readout(x):
    scale = 1.1
    z = scale * x
    return _affine(x + (np.logaddexp(z, -z) - np.log(2.0)) / scale)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    observed = np.concatenate((train['x'], study['test']['x']), axis=0)
    coef = _calibration(study['panel_a'])
    latent = np.linalg.solve(coef[1:].T, (observed - coef[0]).T).T

    panel = study['panel_c']
    readout_1 = _bilinear(latent) @ _fit(_bilinear(panel['u']), panel['v1'])
    readout_2 = _logcosh_readout(latent) @ _fit(_logcosh_readout(panel['u']), panel['v2'])
    h = np.column_stack((readout_1, readout_2))
    return {'h': h, 'n_train': len(train['x']), 'y': train['y'], 'traj': train['traj']}
