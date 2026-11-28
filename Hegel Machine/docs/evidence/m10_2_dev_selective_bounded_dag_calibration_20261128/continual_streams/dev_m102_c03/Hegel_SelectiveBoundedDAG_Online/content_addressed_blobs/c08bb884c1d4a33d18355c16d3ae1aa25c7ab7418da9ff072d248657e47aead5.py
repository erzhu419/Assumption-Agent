import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _coordinates(panel, x):
    """Calibrate and invert the cyclic observation map for the cubic terms."""
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    w = np.roll(u, -1, axis=1)
    linear = np.column_stack((np.ones(len(u)), u))
    r = np.linalg.norm(linear @ _fit(linear, v) - v, axis=1)
    good = r <= 4 * max(np.median(r), 1e-8)
    design = np.stack((np.ones_like(u), u, w, u*w, u**3), axis=1)
    for _ in range(4):
        a = np.stack([_fit(design[good, :, j], v[good, j]) for j in range(4)])
        r = np.linalg.norm(np.einsum('nkj,jk->nj', design, a)-v, axis=1)
        new = r <= 5 * max(np.median(r), 1e-6)
        if new.sum() < 10 or np.array_equal(good, new):
            break
        good = new
    bias, scale, neighbor, mixed, cubic = a.T
    z = (x - bias) / scale
    j = np.arange(z.shape[1])
    for _ in range(14):
        w = np.roll(z, -1, axis=1)
        r = bias + scale*z + neighbor*w + mixed*z*w + cubic*z**3 - x
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = scale + mixed*w + 3*cubic*z*z
        jac[:, j, (j+1) % len(j)] = neighbor + mixed*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -1., 1.)
        if np.max(np.abs(step), initial=0.) < 1e-12:
            break
    return z


def _predict(design, y, traj, times, cubic_count):
    n = len(y)
    train, test = design[:n], design[n:]
    y_hat = test @ _fit(train, y)
    weights = _fit(train, traj)
    if len(times) <= 3:
        return y_hat, test @ weights

    # Fit two saturation rates in the presence of an algebraic growth mode.
    # Profile out the temporal amplitudes before optimizing the rates.
    projected = train[:, :-cubic_count] @ weights[:-cubic_count]
    root = np.sqrt(times)

    def basis(rates):
        return np.column_stack((-np.expm1(-times[:, None]*rates), root))

    def residual(rates):
        q, _ = np.linalg.qr(basis(rates), mode='reduced')
        return (projected - (projected @ q) @ q.T).ravel()

    rates = least_squares(residual, [.42, 1.82], bounds=([.05, .8], [.8, 5.]), max_nfev=100).x
    q, _ = np.linalg.qr(basis(rates), mode='reduced')
    return y_hat, (test @ (weights @ q)) @ q.T


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    features = inputs['node_00']
    train, test = study['train'], study['test']
    x = np.concatenate((train['x'], test['x']), axis=0)
    z = _coordinates(study['panel_a'], x)
    nxt = np.roll(z, -1, axis=1)
    triples = np.column_stack([np.prod(np.delete(z, j, axis=1), axis=1) for j in range(z.shape[1])])
    cubic = np.column_stack((triples, z*z*nxt, z*nxt*nxt))
    design = np.column_stack((features['ht'], z, cubic))
    y_hat, state_pred = _predict(design, np.asarray(train['y']), np.asarray(train['traj']), np.asarray(study['eval_times']), cubic.shape[1])
    end = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
