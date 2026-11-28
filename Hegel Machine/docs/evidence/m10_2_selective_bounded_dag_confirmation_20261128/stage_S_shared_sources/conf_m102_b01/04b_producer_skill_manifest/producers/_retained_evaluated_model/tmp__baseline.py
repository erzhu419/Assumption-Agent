import itertools
import numpy as np


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    offsets, scales = [], []
    for j in range(u.shape[1]):
        d = _aff(u[:, j:j+1]); q = _ls(d, v[:, j])
        for _ in range(8):
            keep = np.argsort(abs(d @ q - v[:, j]))[:int(len(u)*.875)]
            q = _ls(d[keep], v[keep, j])
        err = abs(d @ q - v[:, j]); keep = err <= max(10*np.median(err), 0.002)
        q = _ls(d[keep], v[keep, j])
        offsets.append(q[0]); scales.append(q[1])
    b = study['panel_b']
    return _aff((x - offsets)/scales) @ _ls(_aff(b['u']), b['v'])


def _scalar(w):
    return np.column_stack((_aff(w), w/(1+w*w), w*w, *[w[:, i]*w[:, j] for i,j in itertools.combinations(range(w.shape[1]), 2)]))


def _state(w):
    return np.column_stack((_aff(w), np.arctan(2.5*w)))


def solve_batch(study):
    tr, te = study['train'], study['test']
    n = len(tr['x'])
    w = _operator(study, np.concatenate((tr['x'], te['x']), axis=0))
    f, g = _scalar(w), _state(w)
    y = f[n:] @ _ls(f[:n], tr['y'])
    p = g[n:] @ _ls(g[:n], tr['traj'])
    tail = p[:, -1]
    return {'y_hat': y, 'state_pred': p, 'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
