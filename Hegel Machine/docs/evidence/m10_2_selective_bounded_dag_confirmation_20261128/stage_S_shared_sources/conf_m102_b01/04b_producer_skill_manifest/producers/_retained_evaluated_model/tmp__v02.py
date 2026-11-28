import itertools
import numpy as np
from scipy.optimize import minimize_scalar


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


def _features(w, gain, interaction):
    z = np.arcsinh(gain * w)
    if not interaction:
        return np.column_stack((_aff(w), z))
    pairs = [z[:, i] * z[:, j] for i, j in itertools.combinations_with_replacement(range(w.shape[1]), 2)]
    return np.column_stack((_aff(w), z, *pairs))


def _predict(w, target, interaction):
    n = len(target)
    scale = np.maximum(np.std(target, axis=0), 1e-8)
    def loss(gain):
        f = _features(w[:n], gain, interaction)
        r = (f @ _ls(f, target) - target) / scale
        return np.mean(r*r)
    gain = minimize_scalar(loss, bounds=(0.8, 7), method='bounded').x
    f = _features(w, gain, interaction)
    return f[n:] @ _ls(f[:n], target)


def solve_batch(study):
    tr, te = study['train'], study['test']
    w = _operator(study, np.concatenate((tr['x'], te['x']), axis=0))
    y = _predict(w, tr['y'], True)
    p = _predict(w, tr['traj'], False)
    tail = p[:, -1]
    return {'y_hat': y, 'state_pred': p, 'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
