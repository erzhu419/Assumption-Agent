import numpy as np


def _basis(w):
    r = w * np.roll(w, -1, axis=1)
    z = np.column_stack((w, r))
    return np.column_stack((np.ones(len(w)), z, *(z[:, i]*z[:, j] for i in range(8) for j in range(i,8))))


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    w = inputs['node_00']['w']
    y, tr = train['y'], train['traj']
    n = len(y)
    f = _basis(w)
    scale = np.maximum(f[:n].std(0), 1e-8)
    scale[0] = 1
    f = f/scale
    reg = np.diag(np.r_[0, np.full(f.shape[1]-1, .4)])
    pred = f[n:] @ np.linalg.solve(f[:n].T@f[:n]+reg, f[:n].T@np.column_stack((y,tr)))
    y, tr = pred[:,0], pred[:,1:]
    return {'y_hat': y, 'state_pred': tr, 'diagnostics_vec': np.array([y.mean(), y.std(), tr[:,-1].mean(), tr[:,-1].std()])}
