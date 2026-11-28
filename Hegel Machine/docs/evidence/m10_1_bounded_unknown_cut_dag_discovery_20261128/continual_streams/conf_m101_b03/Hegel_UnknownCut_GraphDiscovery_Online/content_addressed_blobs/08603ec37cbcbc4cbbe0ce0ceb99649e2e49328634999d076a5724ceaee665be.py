import numpy as np


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    data = inputs['node_00']
    z, c, d = data['z'], data['c'], inputs['node_02']['d']
    n = len(train['x'])
    base = np.column_stack((np.ones(len(z)), z, c, d, c * d))
    design = base[:n]
    y_hat = base[n:] @ _fit(design, train['y'])

    # The trajectory needs a cross-coordinate readout product as well.
    # Select it from training residuals after removing the baseline span.
    i, j = np.where(~np.eye(c.shape[1], dtype=bool))
    products = c[:, i] * d[:, j]
    orth = products[:n] - design @ _fit(design, products[:n])
    traj = train['traj']
    residual = traj - design @ _fit(design, traj)
    norms = np.linalg.norm(orth, axis=0)
    gain = np.linalg.norm(orth.T @ residual, axis=1) / np.maximum(norms, 1e-12)
    trajectory_design = np.column_stack((base, products[:, np.argmax(gain)]))
    state_pred = trajectory_design[n:] @ _fit(trajectory_design[:n], traj)
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
