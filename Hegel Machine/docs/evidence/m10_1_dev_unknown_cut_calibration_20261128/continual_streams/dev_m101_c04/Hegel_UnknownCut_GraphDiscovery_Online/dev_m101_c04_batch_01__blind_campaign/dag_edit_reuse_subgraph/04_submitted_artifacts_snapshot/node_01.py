import numpy as np
from scipy.optimize import least_squares


def _scalar_design(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), z, a*c + b*d))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z_train = np.asarray(inputs['node_00']['z_tr'], dtype=float)
    z_test = np.asarray(inputs['node_00']['z_te'], dtype=float)
    train = study['train']
    x = _scalar_design(z_train)
    xt = _scalar_design(z_test)
    y = np.asarray(train['y'], dtype=float)
    initial = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(design, p):
        q = design @ p[:-2]
        return q + p[-2] * np.tanh(p[-1] * q)

    p = least_squares(
        lambda p: predict(x, p) - y,
        np.r_[initial, 0.4, 1.7],
        bounds=(np.r_[np.full(x.shape[1], -np.inf), 0., 0.], np.inf),
    ).x
    y_hat = predict(xt, p)

    weights = np.linalg.lstsq(x[:, :5], train['traj'], rcond=None)[0]
    state_pred = xt[:, :5] @ weights
    terminal = state_pred[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state_pred,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()]),
    }
