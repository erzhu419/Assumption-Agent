import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _cosine_panel(u, v):
    def loss(freq):
        design = _affine(np.cos(freq * u))
        residual = design @ _fit(design, v) - v
        return np.mean(residual ** 2)

    grid = np.linspace(0.2, 6.0, 59)
    k = int(np.argmin([loss(freq) for freq in grid]))
    lo, hi = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    opt = minimize_scalar(loss, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-9})
    freq = min((grid[k], opt.x), key=loss)
    return freq, _fit(_affine(np.cos(freq * u)), v)


def _ring(x):
    return x * np.roll(x, -1, axis=1)


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_affine(x), x[:, i] * x[:, j]))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    w = np.asarray(inputs['node_00']['w'], dtype=float)
    b = study['panel_b']
    # Undo only the affine B handoff, retaining the frozen A calibration.
    affine_b = _fit(_affine(b['u']), b['v'])
    u = _fit(affine_b[1:].T, (w - affine_b[0]).T).T
    freq, cosine_b = _cosine_panel(b['u'], b['v'])
    z = _affine(np.cos(freq * u)) @ cosine_b

    c = study['panel_c']
    ring_c = _fit(_affine(_ring(c['u'])), c['v1'])
    latent = _affine(_ring(z)) @ ring_c
    train = study['train']
    n = len(train['y'])
    scalar_design, traj_design = _quadratic(latent), _affine(latent)
    y = scalar_design[n:] @ _fit(scalar_design[:n], train['y'])
    p = traj_design[n:] @ _fit(traj_design[:n], train['traj'])
    tail = p[:, -1]
    return {'y_hat': y, 'state_pred': p,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
