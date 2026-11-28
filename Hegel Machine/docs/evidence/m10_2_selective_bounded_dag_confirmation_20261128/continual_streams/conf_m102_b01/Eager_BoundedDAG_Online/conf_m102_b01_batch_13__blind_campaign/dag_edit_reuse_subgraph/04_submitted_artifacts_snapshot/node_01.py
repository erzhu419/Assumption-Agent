import itertools
import numpy as np
from scipy.optimize import minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(study, projection):
    b = study['panel_b']
    u, v = b['u'], b['v']
    affine = _ls(_aff(u), v)
    x = _ls(affine[1:].T, (projection - affine[0]).T).T

    def basis(z, frequency):
        return _aff(np.cos(frequency * z))

    def loss(frequency):
        m = basis(u, frequency)
        r = m @ _ls(m, v) - v
        return np.mean(r*r)

    grid = np.linspace(0.4, 8.0, 77)
    i = int(np.argmin([loss(g) for g in grid]))
    g = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                        method='bounded', options={'xatol': 1e-9}).x
    return basis(x, g) @ _ls(basis(u, g), v)


def _features(w, gain, interaction, kind):
    u = gain * w
    if kind == "atan":
        z = np.arctan(u)
    elif kind == "inv3":
        z = np.sinh(np.arcsinh(u) / 3)
    else:
        z = np.arcsinh(u)
    if not interaction:
        return np.column_stack((_aff(w), z))
    pairs = [z[:, i] * z[:, j] for i, j in itertools.combinations_with_replacement(range(w.shape[1]), 2)]
    return np.column_stack((_aff(w), z, *pairs))


def _predict(w, target, interaction, kind):
    n = len(target)
    scale = np.maximum(np.std(target, axis=0), 1e-8)
    def loss(gain):
        f = _features(w[:n], gain, interaction, kind)
        r = (f @ _ls(f, target) - target) / scale
        return np.mean(r*r)
    gain = minimize_scalar(loss, bounds=(0.8, 7), method='bounded').x
    f = _features(w, gain, interaction, kind)
    return f[n:] @ _ls(f[:n], target)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    projection = inputs['node_00']['w']
    tr, te = study['train'], study['test']
    w = _operator(study, projection)
    y = (0.35 * _predict(w, tr['y'], True, 'asinh')
         + 0.65 * _predict(w, tr['y'], True, 'atan'))
    p = _predict(w, tr['traj'], False, 'inv3')
    tail = p[:, -1]
    return {'y_hat': y, 'state_pred': p, 'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
