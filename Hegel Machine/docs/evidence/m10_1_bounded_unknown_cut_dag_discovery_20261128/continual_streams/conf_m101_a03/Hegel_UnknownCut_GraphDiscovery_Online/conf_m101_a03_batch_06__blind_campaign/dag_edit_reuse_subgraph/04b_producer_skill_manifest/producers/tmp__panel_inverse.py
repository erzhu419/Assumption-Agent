import numpy as np


def _features(u):
    n, d = u.shape
    channel = np.broadcast_to(np.eye(d), (n, d, d))
    neighbor = np.roll(u, -1, axis=1)
    return np.concatenate((channel, channel * u[..., None],
                           neighbor[..., None], (u * neighbor)[..., None],
                           (u ** 3)[..., None]), axis=2).reshape(n * d, 2 * d + 3)


def _fit(u, v):
    design = _features(u)
    y = v.ravel()
    d = u.shape[1]
    weights = np.ones(len(u))
    for scale in (0.15, 0.05, 0.015, 0.004, 0.0015, 0.0015):
        w = np.repeat(weights, d)
        p = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
        residual = np.linalg.norm((design @ p).reshape(v.shape) - v, axis=1)
        weights = 1 / np.hypot(1, residual / scale)
    keep = np.repeat(np.argsort(np.argsort(residual)) < len(u) - 3, d)
    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]


def _invert(v, p):
    d = v.shape[1]
    offset, gain = p[:d], p[d:2*d]
    coupling, interaction, cubic = p[-3:]
    j = np.arange(d)
    nxt = (j + 1) % d
    linear = np.diag(gain)
    linear[j, nxt] = coupling
    x = np.linalg.solve(linear, (v - offset).T).T

    def error(x):
        neighbor = x[:, nxt]
        return offset + gain * x + (coupling + interaction * x) * neighbor + cubic * x ** 3 - v

    for _ in range(25):
        r = error(x)
        if np.max(np.abs(r), initial=0.0) < 1e-11:
            break
        jac = np.zeros((len(x), d, d))
        jac[:, j, j] = gain + interaction * x[:, nxt] + 3 * cubic * x ** 2
        jac[:, j, nxt] = coupling + interaction * x
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = x - step
        baseline = np.linalg.norm(r, axis=1)
        for _ in range(12):
            worse = np.linalg.norm(error(trial), axis=1) > baseline
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        x = trial
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    p = _fit(panel['u'], panel['v'])
    return {'x_cal_tr': _invert(study['train']['x'], p),
            'x_cal_te': _invert(study['test']['x'], p)}
