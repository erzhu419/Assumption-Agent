import numpy as np


def _design(z):
    return np.column_stack((np.ones(len(z)), z, *[z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i, z.shape[1])]))


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    stage = inputs['node_00']
    design = _design(stage['c'])
    query = _design(stage['query'])
    y = np.asarray(train['y'], float)
    traj = np.asarray(train['traj'], float)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    return {'y_hat': query @ np.linalg.lstsq(design, y, rcond=None)[0], 'beta': beta, 'fitted': design @ beta, 'new_design': query}
