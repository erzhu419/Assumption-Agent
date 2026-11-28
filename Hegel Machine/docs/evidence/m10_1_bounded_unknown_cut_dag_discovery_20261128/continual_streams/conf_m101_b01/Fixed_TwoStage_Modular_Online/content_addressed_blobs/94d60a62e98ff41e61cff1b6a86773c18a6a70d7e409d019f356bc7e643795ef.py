import numpy as np


def _features(z, products):
    return np.column_stack((np.ones(len(z)), z, products))


def _neighbor_products(z):
    return z*np.roll(z, -1, axis=1)


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    u, z = data['panel_coordinates'], data['latent']
    c1 = np.linalg.lstsq(_features(u, _neighbor_products(u)), data['panel_v1'], rcond=None)[0]
    c2 = np.linalg.lstsq(_features(u, u*u), data['panel_v2'], rcond=None)[0]
    design = _features(z, np.column_stack((
        _features(z, _neighbor_products(z)) @ c1,
        _features(z, z*z) @ c2,
    )))
    target = np.column_stack((data['train_y'], data['train_trajectory']))
    n = len(target)
    coef = np.linalg.lstsq(design[:n], target, rcond=None)[0]
    pred = design[n:] @ coef
    y, state = pred[:, 0], pred[:, 1:]
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
