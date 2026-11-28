import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _mixed(u, z, z_mean):
    products = (u[:, :, None] * (z - z_mean)[:, None, :]).reshape(len(u), -1)
    return np.column_stack((_affine(u), products))


def _scalar(u_tr, z_tr, u_te, z_te, y):
    z_mean = z_tr.mean(axis=0)
    design = _mixed(u_tr, z_tr, z_mean)
    query = _mixed(u_te, z_te, z_mean)
    # The affine mapping makes some mixed products linearly dependent.
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    basis = right[singular > 1e-10 * singular[0]].T
    design, query = design @ basis, query @ basis
    scale = max(2.0, 1.05 * np.max(np.abs(y)))
    initial = np.r_[scale, np.linalg.lstsq(design, y / scale, rcond=None)[0]]

    def residual(params):
        return np.r_[params[0] * np.arctan(design @ params[1:]) - y,
                     0.003 * (params - initial)]

    def jacobian(params):
        q = design @ params[1:]
        derivative = params[0] / (1.0 + q * q)
        return np.vstack((np.column_stack((np.arctan(q), derivative[:, None] * design)),
                          0.003 * np.eye(len(params))))

    fitted = least_squares(residual, initial, jac=jacobian, max_nfev=160).x
    return fitted[0] * np.arctan(query @ fitted[1:])


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    calibrated = inputs['node_00']
    mapped = inputs['node_01']
    u_tr, u_te = calibrated['u_tr'], calibrated['u_te']
    y_hat = _scalar(u_tr, mapped['z_tr'], u_te, mapped['z_te'], train['y'])
    trajectory = np.linalg.lstsq(_affine(u_tr), train['traj'], rcond=None)[0]
    state_pred = _affine(u_te) @ trajectory
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
