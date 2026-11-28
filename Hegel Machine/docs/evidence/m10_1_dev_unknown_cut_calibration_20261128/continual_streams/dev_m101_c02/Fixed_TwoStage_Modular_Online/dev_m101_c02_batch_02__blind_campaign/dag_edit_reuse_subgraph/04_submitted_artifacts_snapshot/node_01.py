import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _scalar_basis(z):
    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def _rational_basis(z, p):
    q = z @ p[:-1] + p[-1]
    denominator = 1.0 + q * q
    return np.column_stack((q, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3])) / denominator[:, None]


def _predict_scalar(z, y, z_test):
    basis = _scalar_basis(z)
    orthogonal, _ = np.linalg.qr(basis, mode='reduced')

    def remove_basis(v):
        return v - orthogonal @ (orthogonal.T @ v)

    target = remove_basis(y)
    direction = np.linalg.lstsq(basis, y, rcond=None)[0][1:5]
    direction /= max(np.linalg.norm(direction), 1e-12)

    # Profile out the linear coefficients while fitting the shared denominator.
    def residual(p):
        terms = remove_basis(_rational_basis(z, p))
        weights = np.linalg.lstsq(terms, target, rcond=None)[0]
        return terms @ weights - target

    fit = min(
        (least_squares(residual, np.r_[scale * direction, bias],
                       bounds=(-5.0, 5.0), max_nfev=300)
         for scale in (0.6, 1.4, 2.5) for bias in (0.0, 0.25, -0.25)),
        key=lambda result: result.cost,
    )
    design = np.column_stack((basis, _rational_basis(z, fit.x)))
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.column_stack((_scalar_basis(z_test), _rational_basis(z_test, fit.x))) @ coeff


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    coordinates = inputs['node_00']
    z, z_test = coordinates['z_tr'], coordinates['z_te']
    y_hat = _predict_scalar(z, train['y'], z_test)
    state_coeff = np.linalg.lstsq(_affine(z), train['traj'], rcond=None)[0]
    state_pred = _affine(z_test) @ state_coeff
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
