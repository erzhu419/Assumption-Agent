import numpy as np
from scipy.optimize import least_squares


def _fit(design, targets):
    return np.linalg.lstsq(design, targets, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _v1_basis(x):
    return _affine(x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]])


def _v2_basis(x):
    return np.column_stack((_affine(x), x * x))


def _relaxation(times, trajectories):
    def basis(rates):
        return -np.expm1(-times[:, None] * np.sort(rates))

    def residual(rates):
        temporal = basis(rates)
        return (temporal @ _fit(temporal, trajectories.T) - trajectories.T).ravel()

    rates = least_squares(residual, [0.41, 1.8], bounds=(0.05, 5.0)).x
    return basis(rates)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    f = inputs['node_00']['f']
    panel = study['panel_c']
    u = panel['u']
    v1 = _v1_basis(x) @ _fit(_v1_basis(u), panel['v1'])
    # Retain the diagonal quadratic approximation for v2 in the sink:
    # it extrapolates better downstream than the closer nonlinear panel fit.
    v2 = _v2_basis(x) @ _fit(_v2_basis(u), panel['v2'])

    left, right = np.triu_indices(f.shape[1])
    pairs = (f[:, :, None] * v2[:, None, :]).reshape(len(f), -1)
    design = np.column_stack((_affine(f), f[:, left] * f[:, right], f * v1, pairs))
    n_train = len(train['x'])
    temporal = _relaxation(np.asarray(study['eval_times']), train['traj'])
    amplitudes = _fit(temporal, train['traj'].T).T
    prediction = design[n_train:] @ _fit(design[:n_train], np.column_stack((train['y'], amplitudes)))
    y_hat = prediction[:, 0]
    state_pred = prediction[:, 1:] @ temporal.T
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
