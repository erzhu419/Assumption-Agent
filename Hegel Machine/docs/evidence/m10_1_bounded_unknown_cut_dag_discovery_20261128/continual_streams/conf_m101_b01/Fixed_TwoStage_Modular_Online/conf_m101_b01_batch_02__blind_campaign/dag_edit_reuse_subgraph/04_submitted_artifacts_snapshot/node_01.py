import numpy as np


def _features(z):
    a, b, c, d = z.T
    return np.column_stack((np.ones(len(z)), a*b, a*d, b*c, c*d))


def _readout_features(base):
    q = base[:, 1:]
    # q0*q3 and q1*q2 are the same monomial; include it only once.
    quadratic = [q[:, i]*q[:, j] for i in range(4) for j in range(i, 4)
                 if (i, j) != (1, 2)]
    return np.column_stack((base, *quadratic))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    train = _features(data['z_tr'])
    test = _features(data['z_te'])
    y_coef = np.linalg.lstsq(_readout_features(train), data['train_y'], rcond=None)[0]
    state_coef = np.linalg.lstsq(train, data['train_traj'], rcond=None)[0]
    y = _readout_features(test) @ y_coef
    state = test @ state_coef
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
