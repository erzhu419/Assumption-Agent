"""Panel-conditioned readout regression and shared relaxation prediction."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _terms(x, degree):
    return np.column_stack([np.prod(x[:, inds], axis=1) for inds in
                            combinations_with_replacement(range(x.shape[1]), degree)])


def _fit(blocks, n, target, ridge):
    design = np.column_stack([features * np.sqrt(weight) /
                              np.maximum(np.sqrt(np.mean(features[:n] ** 2, axis=0)), 1e-10)
                              for features, weight in blocks])
    g = design[:n]
    return (design[n:] @ g.T) @ np.linalg.solve(g @ g.T + ridge * np.eye(n), target)


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    a = _affine(x)
    panel = study['panel_b']
    b = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    g = a @ b
    q = _terms(g, 2)
    yfits = []
    for scale, ridge, qw in ((.7, .02, 1.), (.85, .045, .55), (1., .09, .7), (.7, .02, .15)):
        f = np.sin(scale * g)
        yfits.append(_fit([(a, 1e4), (q, qw), (f, 1.), (_terms(f, 2), 1.)], n, train['y'], ridge))
    y = np.mean(yfits, axis=0)

    t = np.asarray(study['eval_times'])
    measured = np.asarray(train['traj'])
    def shape(rate):
        return -np.expm1(-rate * t)
    def residual(rate):
        profile = shape(rate)
        return np.sum((measured - np.outer(measured @ profile / (profile @ profile), profile))**2)
    profile = shape(minimize_scalar(residual, bounds=(.05, 4.), method='bounded').x)
    amp = measured @ profile / (profile @ profile)
    h = np.linalg.solve(b[1:].T, (x - b[0]).T).T
    pc = study['panel_c']
    transform = lambda v: v + .45*v*v
    cb = np.linalg.lstsq(_affine(transform(pc['u'])), pc['v2'], rcond=None)[0]
    c = _affine(transform(h)) @ cb
    qx = _terms(x, 2)
    cx = _terms(x, 3)
    afits = []
    for coord, scale, ridge, nonlinearity in ((h, 1., .015, np.arctan), (h, 1., .025, np.tanh),
                                                (c, 1.25, .015, np.sin), (c, .85, .007, np.arctan)):
        f = nonlinearity(scale*coord)
        afits.append(_fit([(a, 1e4), (qx, .2), (cx, .02), (f, 1.), (_terms(f, 2), .1)], n, amp, ridge))
    state = np.outer(np.mean(afits, axis=0), profile)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
