import numpy as np
from scipy.optimize import minimize_scalar


def _sensor_fit(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    params = []
    for x, y in zip(u.T, v.T):
        delta = x[j] - x[i]
        spread = np.abs(delta) > 0.15 * np.ptp(x)
        slope = np.median((y[j[spread]] - y[i[spread]]) / delta[spread])
        bias = np.median(y - slope * x)
        residual = y - slope * x - bias
        keep = np.abs(residual) <= max(6 * np.median(np.abs(residual)), 1e-10)
        if np.count_nonzero(keep) >= 2 and np.ptp(x[keep]) > 0:
            bias, slope = np.linalg.lstsq(np.column_stack((np.ones(keep.sum()), x[keep])), y[keep], rcond=None)[0]
        params.append((bias, slope))
    return np.asarray(params).T


def _basis(x, frequency):
    return np.column_stack((np.ones(len(x)), x, np.sin(frequency * x)))


def _operator(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))

    def fit(frequency):
        design = _basis(u, frequency)
        coef = np.linalg.lstsq(design, v, rcond=None)[0]
        return np.sum(np.square(design @ coef - v)), coef

    grid = np.linspace(0.4, 5.5, 31)
    best = np.argmin([fit(f)[0] for f in grid])
    f = minimize_scalar(lambda z: fit(z)[0], bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]), method='bounded').x
    return f, fit(f)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    offset, gain = _sensor_fit(study['panel_a'])
    frequency, coef = _operator(study['panel_b'])

    def coords(x):
        return _basis((np.asarray(x, dtype=float) - offset) / gain, frequency) @ coef

    return {'c': coords(study['train']['x']), 'query': coords(study['test']['x'])}
