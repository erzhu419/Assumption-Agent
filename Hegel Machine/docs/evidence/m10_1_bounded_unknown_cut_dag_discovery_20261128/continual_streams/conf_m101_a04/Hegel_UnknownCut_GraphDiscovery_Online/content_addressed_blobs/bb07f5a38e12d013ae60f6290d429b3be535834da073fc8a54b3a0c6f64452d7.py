import numpy as np


def _basis(x, scale):
    even = np.logaddexp(scale * x, -scale * x) - np.log(2.0)
    return np.column_stack((np.ones(len(x)), x, even))


def _fit(u, v):
    # Profile out the linear coefficients; all channels share one scale.
    def fit_at(scale):
        basis = _basis(u, scale)
        coef = np.linalg.lstsq(basis, v, rcond=None)[0]
        return np.sum((basis @ coef - v)**2), coef

    grid = np.geomspace(0.1, 4.0, 17)
    i = min(range(len(grid)), key=lambda j: fit_at(grid[j])[0])
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    for _ in range(28):
        a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
        if fit_at(a)[0] <= fit_at(b)[0]:
            hi = b
        else:
            lo = a
    scale = (lo + hi) / 2
    return scale, fit_at(scale)[1]


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    scale, coef = _fit(panel['u'], panel['v2'])
    return {'d': _basis(panel['x'], scale) @ coef}
