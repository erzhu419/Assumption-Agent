import numpy as np
from scipy.optimize import minimize_scalar


def _aff(z):
    return np.column_stack((np.ones(len(z)), z))


def _ls(a, y):
    return np.linalg.lstsq(a, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    z = np.empty_like(x, dtype=float)
    for j in range(u.shape[1]):
        a, y = _aff(u[:, j]), v[:, j]
        q = _ls(a, y)
        k = max(2, int(.875 * len(y)))
        for _ in range(8):
            keep = np.argsort(np.abs(a @ q - y))[:k]
            q = _ls(a[keep], y[keep])
        err = np.abs(a @ q - y)
        keep = err <= max(10 * np.median(err), .002)
        if np.count_nonzero(keep) >= 2:
            q = _ls(a[keep], y[keep])
        z[:, j] = (x[:, j] - q[0]) / q[1]
    return z


def _panel_b(panel, z):
    u, v = panel['u'], panel['v']

    def design(x, freq):
        return _aff(np.cos(freq * x))

    def loss(freq):
        a = design(u, freq)
        return np.mean((a @ _ls(a, v) - v) ** 2)

    # Profile out the affine mixing while estimating the common even warp.
    grid = np.linspace(.25, 8., 64)
    i = int(np.argmin([loss(freq) for freq in grid]))
    lo, hi = grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]
    fit = minimize_scalar(loss, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-9})
    freq = min((grid[i], fit.x), key=loss)
    return design(z, freq) @ _ls(design(u, freq), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _calibrate(study['panel_a'], x)
    return {'w': _panel_b(study['panel_b'], z)}
