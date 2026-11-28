import numpy as np


def _panel_features(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _fit_predict(design, target):
    n = len(target)
    return design[n:] @ np.linalg.lstsq(design[:n], target, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    panel = study['panel_c']
    panel_map = np.linalg.lstsq(_panel_features(panel['u']), panel['v1'], rcond=None)[0]
    v1 = _panel_features(inputs['node_01']['x']) @ panel_map
    affine = np.column_stack((np.ones(len(v1)), v1))
    a, b, c, d = v1.T
    readout = np.column_stack((affine, a*c, b*d))
    y = _fit_predict(readout, train['y'])
    state = _fit_predict(affine, train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
