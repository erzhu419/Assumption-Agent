"""Calibrate the panels and construct bilinear response features."""
from scipy.optimize import minimize_scalar
import numpy as np

def _affine_design(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibrate(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    corrected = np.empty_like(measured, dtype=float)
    for j in range(u.shape[1]):
        d = _affine_design(u[:, j:j+1])
        keep = np.ones(len(u), bool)
        for _ in range(8):
            coef = np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
            residual = np.abs(d @ coef - v[:, j])
            new = residual <= max(.0005, 4 * np.median(residual))
            if new.sum() < 3 or np.array_equal(new, keep):
                break
            keep = new
        corrected[:, j] = (measured[:, j] - coef[0]) / coef[1]
    return corrected


def _panel_c_basis(q, rate):
    return np.column_stack((np.ones(len(q)), q, q**3 / (1 + rate*q*q)))


def _panel_c_coordinates(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1'))

    def fit(log_rate):
        d = _panel_c_basis(u, np.exp(log_rate))
        coef = np.linalg.lstsq(d, v, rcond=None)[0]
        return np.sum((d @ coef - v)**2), coef

    log_rate = minimize_scalar(lambda r: fit(r)[0], bounds=(-5., 3.),
                               method='bounded').x
    return _panel_c_basis(z, np.exp(log_rate)) @ fit(log_rate)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    measured = np.vstack((train, study['test']['x']))
    x = _calibrate(study['panel_a'], measured)
    panel = study['panel_b']
    b = np.linalg.lstsq(_affine_design(panel['u']), panel['v'], rcond=None)[0]
    w = _panel_c_coordinates(study['panel_c'], _affine_design(x) @ b)
    interactions = (x[:, :, None] * w[:, None, :]).reshape(len(x), -1)
    corrected = np.column_stack((x, w, interactions))
    # The frozen basis passes all coordinates through linearly but only
    # expands slots 0:4. Leave that block inactive: the required products
    # are between calibrated A and C coordinates, not within either set.
    operator = np.zeros((corrected.shape[1] + 1, corrected.shape[1] + 4))
    operator[1:, 4:] = np.eye(corrected.shape[1])
    return dict(corrected=corrected, operator=operator, n=np.array(len(train)))
