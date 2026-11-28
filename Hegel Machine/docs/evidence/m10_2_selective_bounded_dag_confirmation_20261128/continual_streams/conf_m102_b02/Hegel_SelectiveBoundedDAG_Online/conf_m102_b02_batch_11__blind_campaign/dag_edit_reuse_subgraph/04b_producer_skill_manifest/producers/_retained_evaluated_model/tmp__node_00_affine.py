"""Calibrate the observations and fit the current panel-B affine map."""
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


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    measured = np.vstack((train, study['test']['x']))
    corrected = _calibrate(study['panel_a'], measured)
    panel = study['panel_b']
    operator = np.linalg.lstsq(_affine_design(panel['u']), panel['v'], rcond=None)[0]
    return dict(corrected=corrected, operator=operator, n=np.array(len(train)))
