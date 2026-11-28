import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, trim=0):
    design, values = _affine(panel['u']), panel['v']
    keep = np.arange(len(design))
    for _ in range(5 if trim else 0):
        coef = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(design @ coef - values, axis=1))[:-trim]
    return np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]


def _harmonics(x, frequency):
    phase = x * frequency
    return np.column_stack((_affine(x), np.sin(phase), np.cos(phase)))


def _panel_frequencies(panel):
    u, values = panel['u'], panel['v1']

    def residual(frequency):
        design = _harmonics(u, frequency)
        coef = np.linalg.lstsq(design, values, rcond=None)[0]
        return (design @ coef - values).ravel()

    fits = (least_squares(residual, np.full(4, start), bounds=(0.15, 8),
                          max_nfev=150) for start in (1.5, 2.5, 3.5, 4.5, 5.5))
    return min(fits, key=lambda fit: fit.cost).x


def _kernel(x, u, w, v, harmonics, reference):
    dot = x @ u.T
    quadratic = (dot**2 + (x*x) @ (u*u).T) / 2
    distance = np.sum((w[:, None] - v[None, :])**2, axis=-1)
    return 1 + dot + quadratic + np.exp(-distance) + 0.1 * (harmonics @ reference.T)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    shared = inputs['node_00']
    x, xt = study['train']['x'], study['test']['x']
    # Restore panel coordinates: the harmonic frequencies are in panel units.
    a, b = _calibration(study['panel_a'], 3), _calibration(study['panel_b'])
    calibrated = _affine(_affine(x) @ a) @ b
    scale, center = calibrated.std(0), calibrated.mean(0)
    w = shared['z'] * scale + center
    wt = shared['zt'] * scale + center
    frequency = _panel_frequencies(study['panel_c'])
    h, ht = _harmonics(w, frequency), _harmonics(wt, frequency)
    gram = _kernel(x, x, w, w, h, h)
    gram.flat[::len(x)+1] += 0.002
    weights = np.linalg.solve(gram, study['train']['y'])
    return {'y_hat': _kernel(xt, x, wt, w, ht, h) @ weights}
