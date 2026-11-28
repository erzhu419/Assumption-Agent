"""Infer the calibration, cyclic readout, and shared trajectory modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _cal_design(x):
    v = np.roll(x, -1, axis=1)
    return np.stack((np.ones_like(x), x, v, x*v, x**3), axis=-1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _cal_design(u)
    good = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])
        e = np.linalg.norm(np.einsum('nki,ki->nk', d, w)-v, axis=1)
        good = e <= np.sort(e)[int(.79*len(u))]
    good = e < 12*np.median(e)
    return np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])


def _inverse_calibration(measured, w):
    x = (measured-w[:, 0])/w[:, 1]
    j = np.arange(x.shape[1])
    for _ in range(18):
        nxt = np.roll(x, -1, axis=1)
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = w[:, 1]+w[:, 3]*nxt+3*w[:, 4]*x*x
        jac[:, j, (j+1)%len(j)] = w[:, 2]+w[:, 3]*x
        res = np.einsum('nki,ki->nk', _cal_design(x), w)-measured
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        x -= step
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return x


def _branches(study, raw):
    x = _inverse_calibration(raw, _calibration(study['panel_a']))
    axle = study['panel_b']
    def bdesign(x):
        return _affine(x**3 / np.sqrt(1 + .70020846*x**4) - .08899942*x)
    b = bdesign(x) @ _fit(bdesign(axle['u']), axle['v'])
    c = study['panel_c']
    def c1design(v):
        return _affine(v * np.roll(v, -1, axis=1))
    return x, b, c1design(b) @ _fit(c1design(c['u']), c['v1'])


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x, b, c = _branches(study, np.r_[train['x'], study['test']['x']])
    # The shared cyclic interaction is read out independently for y and each time.
    design = np.c_[_affine(x), b, c, x * np.roll(c, 1, axis=1)]
    pred = design[n:] @ _fit(design[:n], np.c_[train['y'], train['traj']])
    y, state = pred[:, 0], pred[:, 1:]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
