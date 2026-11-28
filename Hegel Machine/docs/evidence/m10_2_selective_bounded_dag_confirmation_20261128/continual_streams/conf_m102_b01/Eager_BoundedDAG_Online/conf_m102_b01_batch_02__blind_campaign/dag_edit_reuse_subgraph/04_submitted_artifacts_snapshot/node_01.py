import numpy as np


def _pairs(x):
    return np.column_stack([x[:, i] * x[:, j]
                            for i in range(x.shape[1])
                            for j in range(i + 1, x.shape[1])])


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    w = np.asarray(inputs['node_00']['w'], dtype=float)
    train = study['train']
    panel = study['panel_c']
    n = len(train['y'])

    # Identify the calibrated quadratic coordinates using the reference panel.
    ref = _pairs(np.asarray(panel['u'], dtype=float))
    v1 = np.asarray(panel['v1'], dtype=float)
    strength = np.linalg.norm(_fit(_affine(ref), v1)[1:], axis=1)
    keep = strength > 0.01 * strength.max()
    raw_pairs = _pairs(w)
    z = _affine(raw_pairs[:, keep]) @ _fit(_affine(ref[:, keep]), v1)

    # The scalar response couples pairs of panel coordinates; the trajectory
    # is affine in those coordinates at each sampled time.
    scalar_basis = np.column_stack((_affine(raw_pairs), _pairs(z)))
    trajectory_basis = _affine(z)
    y = scalar_basis[n:] @ _fit(scalar_basis[:n], train['y'])
    state = trajectory_basis[n:] @ _fit(trajectory_basis[:n], train['traj'])
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
