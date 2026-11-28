import numpy as np


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _affine_calibration(panel):
    u, v = panel['u'], panel['v']
    design = np.column_stack((np.ones(len(u)), u))
    coef = _fit(design, v)
    # Fit in the forward direction so corrupted measurements are residuals,
    # not high-leverage predictors. Trim the three identified outlier rows.
    for _ in range(10):
        residual = np.linalg.norm(v - design @ coef, axis=1)
        weight = np.sqrt(np.minimum(1.0, 1.5 * max(np.median(residual), 1e-15)
                                    / np.maximum(residual, 1e-15)))
        coef = _fit(design * weight[:, None], v * weight[:, None])
    residual = np.linalg.norm(v - design @ coef, axis=1)
    keep = np.argsort(residual)[:-3]
    return _fit(design[keep], v[keep])


def _pair_basis(u):
    a, b, c, d = u.T
    return np.column_stack((np.ones(len(u)), a*b, a*d, b*c, c*d))


def _cosine_basis(u):
    return np.column_stack((np.ones(len(u)), np.cos(2.4 * u)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    x = np.concatenate((train, test), axis=0)
    affine = _affine_calibration(study['panel_a'])
    u = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T

    panel = study['panel_c']
    v1 = _pair_basis(u) @ _fit(_pair_basis(panel['u']), panel['v1'])
    v2 = _cosine_basis(u) @ _fit(_cosine_basis(panel['u']), panel['v2'])
    interaction = (v1[:, :, None] * v2[:, None, :]).reshape(len(u), -1)
    design = np.column_stack((v1, v2, interaction))
    return {'design': design, 'n_train': np.asarray(len(train))}
