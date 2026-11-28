"""Calibrated bilinear readout and reduced-rank trajectory prediction."""
import numpy as np


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = _affine(u)
    coef = _fit(a, v)
    for _ in range(24):
        residual = np.linalg.norm(a @ coef - v, axis=1)
        scale = max(float(np.median(residual)), 1e-12)
        weight = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(residual, 1e-12)))
        coef = _fit(a * weight[:, None], v * weight[:, None])
    residual = np.linalg.norm(a @ coef - v, axis=1)
    good = residual <= 5 * max(float(np.median(residual)), 1e-12)
    if good.sum() >= a.shape[1] and np.linalg.matrix_rank(a[good]) == a.shape[1]:
        coef = _fit(a[good], v[good])
        mat = coef[1:]
        off = mat - np.diag(np.diag(mat))
        if np.linalg.norm(off) <= 5 * max(float(np.median(residual)), 1e-12):
            for j in range(u.shape[1]):
                coef[0, j], coef[j + 1, j] = _fit(a[good][:, [0, j + 1]], v[good, j])
            coef[1:] = np.diag(np.diag(coef[1:]))
    return coef


def _readout(z):
    return np.column_stack((z, z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    target = np.asarray(study['test']['x'], dtype=float)
    a = _calibration(study['panel_a'])
    b = _fit(_affine(study['panel_b']['u']), study['panel_b']['v'])

    def encode(rows):
        calibrated = np.linalg.solve(a[1:].T, (rows - a[0]).T).T
        return _affine(calibrated) @ b

    z, zt = encode(x), encode(target)
    yh = _readout(zt) @ _fit(_readout(z), train['y'])
    q, r = np.linalg.qr(z, mode='reduced')
    u, sv, vh = np.linalg.svd(q.T @ train['traj'], full_matrices=False)
    rank = min(2, len(sv))
    coef = _fit(r, (u[:, :rank] * sv[:rank]) @ vh[:rank])
    sp = zt @ coef
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
