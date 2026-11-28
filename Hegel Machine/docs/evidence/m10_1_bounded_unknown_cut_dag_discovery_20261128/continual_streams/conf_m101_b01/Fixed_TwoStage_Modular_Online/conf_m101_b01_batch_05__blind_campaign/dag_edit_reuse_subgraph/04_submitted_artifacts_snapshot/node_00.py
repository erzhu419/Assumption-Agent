import numpy as np


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _align(panel, measured):
    u, v = panel['u'], panel['v']
    rows, dims = u.shape
    i = np.arange(dims)
    design = np.zeros((rows, dims, 2 + 2*dims))
    design[:, :, 0] = np.roll(u, -1, axis=1) * (u - 1)
    design[:, :, 1] = u**3
    design[:, i, 2+i] = 1
    design[:, i, 2+dims+i] = u
    flat = design.reshape(rows*dims, -1)
    w = np.ones(rows)
    for _ in range(20):
        sw = np.repeat(np.sqrt(w), dims)
        coef = _fit(flat*sw[:, None], v.ravel()*sw)
        error = np.linalg.norm(design @ coef - v, axis=1)
        scale = max(2*np.median(error), 1e-8)
        w = 1/(1 + (error/scale)**2)
    keep = np.argsort(error)[:-3]
    g, d, *rest = _fit(design[keep].reshape(len(keep)*dims, -1), v[keep].ravel())
    offset, slope = np.array(rest).reshape(2, dims)
    z = (measured-offset)/slope
    jac = np.zeros((len(z), dims, dims))
    for _ in range(24):
        nxt = np.roll(z, -1, axis=1)
        residual = offset + slope*z + g*nxt*(z-1) + d*z**3 - measured
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, i, i] = slope + g*nxt + 3*d*z*z
        jac[:, i, (i+1) % dims] = g*(z-1)
        z -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -.45, .45)
    return z


def _b_features(u, freq):
    return np.column_stack((np.ones(len(u)), u, np.sin(u*freq), u*u))


def _activate(panel, z):
    u, v = panel['u'], panel['v']

    def loss(freq):
        design = _b_features(u, freq)
        error = design @ _fit(design, v) - v
        return np.sum(error*error)

    grid = np.linspace(2.0, 3.6, 33)
    freq = np.full(u.shape[1], min(grid, key=loss))
    for step in (.08, .04, .02, .01, .005, .0025):
        for _ in range(3):
            for j in range(len(freq)):
                tests = np.repeat(freq[None, :], 3, axis=0)
                tests[:, j] += (0, -step, step)
                freq = min(tests, key=loss)
    return _b_features(z, freq) @ _fit(_b_features(u, freq), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    x = np.concatenate((train['x'], test['x']))
    latent = _activate(study['panel_b'], _align(study['panel_a'], x))
    panel = study['panel_c']
    return {'latent': latent, 'panel_coordinates': panel['u'],
            'panel_v1': panel['v1'], 'panel_v2': panel['v2'],
            'train_y': train['y'], 'train_trajectory': train['traj']}
