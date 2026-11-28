import numpy as np
from scipy.optimize import least_squares


def _linear_fit(design, response):
    return np.linalg.lstsq(design, response, rcond=1e-9)[0]


def _affine_design(x):
    return np.column_stack((np.ones(len(x)), x))


def _align(panel):
    readout = np.asarray(panel['v'], dtype=float)
    reference = np.asarray(panel['u'], dtype=float)
    pairs = np.triu_indices(len(reference), 1)
    coefficients = []
    for x, y in zip(reference.T, readout.T):
        dx = (x[:, None] - x)[pairs]
        dy = (y[:, None] - y)[pairs]
        valid = np.abs(dx) > 1e-8
        slope = np.median(dy[valid] / dx[valid])
        offset = np.median(y - slope * x)
        design = _affine_design(x)
        for _ in range(5):
            error = np.abs(y - offset - slope * x)
            keep = error <= max(5 * np.median(error), 0.00035)
            offset, slope = _linear_fit(design[keep], y[keep])
        coefficients.append((offset, slope))
    return np.asarray(coefficients).T


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_affine_design(x), x[:, i] * x[:, j]))


def _curvature(x, q):
    xx = x * x
    return np.column_stack((_affine_design(x), xx, xx / (1 + q[0] * xx)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train_x = np.asarray(study['train']['x'], dtype=float)
    test_x = np.asarray(study['test']['x'], dtype=float)
    n_train = len(train_x)
    offset, slope = _align(study['panel_a'])
    x = (np.vstack((train_x, test_x)) - offset) / slope
    panel = study['panel_c']
    u = np.asarray(panel['u'], dtype=float)
    v1 = np.asarray(panel['v1'], dtype=float)
    v2 = np.asarray(panel['v2'], dtype=float)
    quad = _quadratic(x) @ _linear_fit(_quadratic(u), v1)

    def residual(q):
        f = _curvature(u, q)
        return (f @ _linear_fit(f, v2) - v2).ravel()

    q = least_squares(residual, [0.25], bounds=([0.08], [0.65]), gtol=1e-13).x
    curve = _curvature(x, q) @ _linear_fit(_curvature(u, q), v2)
    features = np.column_stack((quad, curve))
    training = features[:n_train]
    features = 0.6 * (features - training.mean(axis=0)) / np.maximum(training.std(axis=0), 0.1)
    return {'features': features, 'n_train': n_train}
