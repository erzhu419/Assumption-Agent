import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _align(panel):
    design, measured = _affine(panel['u']), panel['v']
    weights = np.ones(len(design))
    for _ in range(20):
        scale = np.sqrt(weights)[:, None]
        coeff = _fit(design * scale, measured * scale)
        distance = np.linalg.norm(measured - design @ coeff, axis=1)
        weights = np.minimum(1., 1.5 * np.median(distance) / np.maximum(distance, 1e-12))
    # Refit without the three documented corrupted calibration rows.
    inliers = np.argsort(distance)[:-3]
    return _fit(design[inliers], measured[inliers])


def _mixer_basis(u, frequency):
    return np.column_stack((_affine(u), np.sin(frequency * u)))


def _mixer(panel):
    u, v = panel['u'], panel['v']

    def loss(frequency):
        design = _mixer_basis(u, frequency)
        return np.sum((v - design @ _fit(design, v)) ** 2)

    grid = np.linspace(0.5, 5., 19)
    i = min(range(len(grid)), key=lambda j: loss(grid[j]))
    freq = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                           method='bounded').x
    return freq, _fit(_mixer_basis(u, freq), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    alignment = _align(study['panel_a'])
    freq, coeff = _mixer(study['panel_b'])

    def encode(x):
        u = np.linalg.solve(alignment[1:].T, (x - alignment[0]).T).T
        return _mixer_basis(u, freq) @ coeff

    return {'z_tr': encode(study['train']['x']),
            'z_te': encode(study['test']['x'])}
