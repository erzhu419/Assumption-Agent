import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _pairs(z):
    return z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]]


def _panel_features(study, x):
    panel = study['panel_a']
    u, v = panel['u'], panel['v']
    keep = np.max(np.abs(u - v), axis=1) < 1
    cal = np.array([np.linalg.lstsq(_affine(u[keep, j]), v[keep, j], rcond=None)[0]
                    for j in range(u.shape[1])])
    z = (x - cal[:, 0]) / cal[:, 1]

    panel = study['panel_c']
    u, v1, v2 = panel['u'], panel['v1'], panel['v2']
    pair_coef = np.linalg.lstsq(_affine(_pairs(u)), v1, rcond=None)[0]

    def cosine_fit(freq):
        ref = _affine(np.cos(freq * u))
        coef = np.linalg.lstsq(ref, v2, rcond=None)[0]
        return np.mean((ref @ coef - v2)**2), coef

    freq = minimize_scalar(lambda k: cosine_fit(k)[0], bounds=(1.6, 3.2),
                           method='bounded').x
    c = _affine(_pairs(z)) @ pair_coef
    d = _affine(np.cos(freq * z)) @ cosine_fit(freq)[1]
    return np.column_stack((c, d, (c[:, :, None] * d[:, None, :]).reshape(len(x), -1)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    sparse = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    f = _panel_features(study, np.vstack((train['x'], study['test']['x'])))
    f = (f - f[:n].mean(axis=0)) / np.maximum(f[:n].std(axis=0), 1e-9)

    # Regress the scalar and two shared relaxation amplitudes on the panel features.
    times = np.asarray(study['eval_times'])
    modes = -np.expm1(-times[:, None] * np.array([0.425, 1.832]))
    amplitude = np.linalg.lstsq(modes, train['traj'].T, rcond=None)[0].T
    target = np.column_stack((train['y'], amplitude))
    mean = target.mean(axis=0)
    ridge = np.sqrt(1e-5 * f.shape[1]) * np.eye(f.shape[1])
    coef = np.linalg.lstsq(np.vstack((f[:n], ridge)),
                           np.vstack((target - mean, np.zeros((f.shape[1], 3)))),
                           rcond=None)[0]
    pred = f[n:] @ coef + mean

    # Express the calibrated readout in the frozen sink's panel coordinates.
    y = (pred[:, 0] - 0.5 * sparse['y_hat']) / 0.5
    traj = (pred[:, 1:] @ modes.T - 0.35 * sparse['state_pred']) / 0.65
    return {'panel_pred': np.column_stack((y, traj))}
