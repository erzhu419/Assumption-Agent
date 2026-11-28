import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _products(x, y):
    return (x[:, :, None] * y[:, None, :]).reshape(len(x), x.shape[1] * y.shape[1])


def _cross_pairs(x):
    i, j = (0, 0, 1, 2), (1, 3, 2, 3)
    return _affine(x[:, i] * x[:, j])


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_affine(x), x[:, i] * x[:, j]))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _features(study, x):
    a = study['panel_a']
    d = _affine(a['u'])
    keep = np.linalg.norm(d @ _fit(d, a['v']) - a['v'], axis=1) < 3
    calibration = np.column_stack([
        _fit(d[keep][:, [0, i + 1]], a['v'][keep, i])
        for i in range(x.shape[1])
    ])
    z = (x - calibration[0]) / calibration[1]
    c = study['panel_c']
    v1 = _cross_pairs(z) @ _fit(_cross_pairs(c['u']), c['v1'])
    v2 = _affine(np.cos(2.4 * z)) @ _fit(_affine(np.cos(2.4 * c['u'])), c['v2'])
    return np.column_stack((_quadratic(v1), _quadratic(v2)[:, 1:], _products(v1, v2)))


def _temporal_modes(times, traj):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        q = basis(rates)
        return (q @ _fit(q, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [.4, 1.85], bounds=([.1, .9], [.9, 4.]), max_nfev=80).x
    return basis(rates)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    upstream = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    features = _features(study, np.vstack((train['x'], study['test']['x'])))
    modes = _temporal_modes(np.asarray(study['eval_times']), train['traj'])
    targets = np.column_stack((train['y'], _fit(modes, train['traj'].T).T))
    pred = features[n:] @ _fit(features[:n], targets)
    # Express both readouts in the unchanged sink's blend coordinates.
    y_panel = (pred[:, 0] - .5 * upstream['y_hat']) / .5
    state_panel = (pred[:, 1:] @ modes.T - .35 * upstream['state_pred']) / .65
    return {'panel_pred': np.column_stack((y_panel, state_panel))}
