def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_features(x, j):
    p, q = x[:, j], x[:, (j + 1) % 4]
    return _affine(np.column_stack((p, q, p * q, p**3)))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    design = [_calibration_features(u, j) for j in range(4)]
    rng = np.random.default_rng(37)
    best = np.inf
    for _ in range(48):
        rows = rng.choice(len(u), 8, replace=False)
        fit = np.column_stack([a @ _ls(a[rows], v[rows, j]) for j, a in enumerate(design)])
        errors = np.linalg.norm(v - fit, axis=1)
        score = np.median(errors)
        if score < best:
            best = score
            keep = errors <= 5 * score
    return np.array([_ls(a[keep], v[keep, j]) for j, a in enumerate(design)])


def _unwarp(x, coeff):
    offset, gain, neighbor, cross, cubic = coeff.T
    j = np.arange(4)
    k = (j + 1) % 4
    linear = np.diag(gain)
    linear[j, k] = neighbor
    u = np.linalg.solve(linear, (x - offset).T).T
    jac = np.zeros((len(x), 4, 4))
    for _ in range(20):
        q = u[:, k]
        residual = offset + gain*u + neighbor*q + cross*u*q + cubic*u**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, j, j] = gain + cross*q + 3*cubic*u**2
        jac[:, j, k] = neighbor + cross*u
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        u -= delta / np.maximum(1, np.linalg.norm(delta, axis=1, keepdims=True))
    return u


def _coordinates(study):
    coeff = _calibrate(study['panel_a'])
    b = study['panel_b']
    mixing = _ls(_affine(b['u']), b['v'])
    return tuple(_affine(_unwarp(study[k]['x'], coeff)) @ mixing for k in ('train', 'test'))


