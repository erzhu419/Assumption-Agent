import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _align(panel, x):
    design, v = _affine(panel['u']), panel['v']
    coef = _fit(design, v)
    for _ in range(12):
        residual = np.linalg.norm(v - design @ coef, axis=1)
        scale = max(3 * np.median(residual), np.finfo(float).eps)
        weight = 1 / np.hypot(1, residual / scale)
        coef = _fit(design * weight[:, None], v * weight[:, None])
    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T


def _cycle(x):
    return x * np.roll(x, -1, axis=1)


def _b_basis(x, frequency):
    return np.column_stack((_affine(x), np.sin(frequency * x)))


def _c2_basis(x):
    return np.column_stack((_affine(x), x*x, np.cos(2.6*x) - 1))


def _panel_b(panel, x, unmix=False):
    u, v = panel['u'], panel['v']
    def loss(f):
        d = _b_basis(u, f)
        return np.linalg.norm(v - d @ _fit(d, v))
    grid = np.linspace(0.5, 7.0, 66)
    best = grid[np.argmin([loss(f) for f in grid])]
    left, right = best - .1, best + .1
    for _ in range(20):
        a, b = (2*left + right)/3, (left + 2*right)/3
        if loss(a) < loss(b):
            right = b
        else:
            left = a
    f = (left + right)/2
    coef = _fit(_b_basis(u, f), v)
    b = _b_basis(x, f) @ coef
    if unmix:
        h = np.linalg.solve(coef[1:5].T, (b - coef[0]).T).T
        return b, h
    return b


def _features(s, x, unmix=False):
    z = _align(s['panel_a'], x)
    b = _panel_b(s['panel_b'], z, unmix)
    c = s['panel_c']
    c1 = _affine(_cycle(z)) @ _fit(_affine(_cycle(c['u'])), c['v1'])
    c2 = _c2_basis(z) @ _fit(_c2_basis(c['u']), c['v2'])
    return z, b, c1, c2


def _polynomial_features(blocks, n, degree=4, rate=.15, scale=3.):
    # Factor the centered polynomial kernel in the training-row span.
    design = []
    for x, weight in zip(blocks, (.25, .50, .25)):
        x = x - x[:n].mean(axis=0)
        energy = np.sum(x[:n] ** 2) / n
        design.append(x * np.sqrt(weight / max(energy, np.finfo(float).eps)))
    x = np.column_stack(design)
    k = (1 + rate * (x @ x[:n].T)) ** degree
    row_mean = k.mean(axis=1)
    train_mean = row_mean[:n]
    k = k - row_mean[:, None] - train_mean[None, :] + train_mean.mean()
    value, vector = np.linalg.eigh(k[:n])
    keep = value > value[-1] * 1e-12
    mapped = k @ (vector[:, keep] * np.sqrt(scale / value[keep]))
    return np.column_stack((np.ones(len(x)), mapped))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    train = s['train']
    x = np.vstack((train['x'], s['test']['x']))
    z, (b, h), c1, c2 = _features(s, x, unmix=True)
    return {'g': _polynomial_features((z, b, h), len(train['x'])),
            'fd': b, 'cd': np.column_stack((c1, c2)),
            'y': train['y'], 'traj': train['traj']}
