import numpy as np


# Panel A couples each coordinate to its cyclic successor.
_NEXT = np.array([1, 2, 3, 0])
_IDX = np.arange(4)


def _fit_panel_a(u, v):
    affine = np.column_stack((np.ones(len(u)), u))
    pair = u * u[:, _NEXT]
    cube = u ** 3
    designs = [np.column_stack((affine, pair[:, j], cube[:, j])) for j in _IDX]
    weight = np.ones(len(u))

    def fit():
        return np.column_stack([
            np.linalg.lstsq(d * weight[:, None], v[:, j] * weight, rcond=None)[0]
            for j, d in enumerate(designs)
        ])

    for _ in range(25):
        coef = fit()
        predicted = np.column_stack([d @ coef[:, j] for j, d in enumerate(designs)])
        residual = np.linalg.norm(predicted - v, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / scale)
    weight[np.argsort(residual)[-3:]] = 0
    return fit()


def _a_forward(u, coef):
    return coef[0] + u @ coef[1:5] + coef[5] * u * u[:, _NEXT] + coef[6] * u ** 3


def _panel_a(panel, observed):
    coef = _fit_panel_a(panel['u'], panel['v'])
    u = np.linalg.solve(coef[1:5].T, (observed - coef[0]).T).T
    for _ in range(30):
        residual = _a_forward(u, coef) - observed
        error = np.linalg.norm(residual, axis=1)
        if np.max(error) < 1e-11:
            break
        jac = np.broadcast_to(coef[1:5].T, (len(u), 4, 4)).copy()
        jac[:, _IDX, _IDX] += coef[5] * u[:, _NEXT] + 3 * coef[6] * u ** 2
        jac[:, _IDX, _NEXT] += coef[5] * u
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        trial = u - step
        for _ in range(12):
            worse = np.linalg.norm(_a_forward(trial, coef) - observed, axis=1) > error
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = u[worse] - step[worse]
        u = trial
    return u


def _panel_c_v2(panel, calibrated):
    def design(u):
        return np.column_stack((np.ones(len(u)), np.cos(2.4 * u)))

    coef = np.linalg.lstsq(design(panel['u']), panel['v2'], rcond=None)[0]
    return design(calibrated) @ coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    calibrated = _panel_a(study['panel_a'], np.vstack((train['x'], test['x'])))
    return {
        'z': np.column_stack((calibrated, _panel_c_v2(study['panel_c'], calibrated))),
        'n_train': np.asarray(len(train['x'])),
        'eval_times': np.asarray(study['eval_times']),
        'y': np.asarray(train['y']),
        'traj': np.asarray(train['traj']),
    }
