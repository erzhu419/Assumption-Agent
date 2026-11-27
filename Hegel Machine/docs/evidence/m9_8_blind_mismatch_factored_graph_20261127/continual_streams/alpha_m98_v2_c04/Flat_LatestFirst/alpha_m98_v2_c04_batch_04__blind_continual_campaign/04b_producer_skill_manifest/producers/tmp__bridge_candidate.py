import numpy as np

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _bridge_basis(x):
    return np.column_stack((np.ones(len(x)), x, x**3,
                            *(x[:, i]*x[:, j] for i, j in _PAIRS)))


def _fit_bridge(cal):
    x = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    a = _bridge_basis(x)
    # Find the ordinary forward response despite large discordant sweep readings.
    mag = np.max(np.abs(raw - np.median(raw, axis=0)), axis=1)
    keep = mag < 5*np.median(mag)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]
        err = np.linalg.norm(a@coef-raw, axis=1)
        keep = err < 8*max(np.median(err), 1e-8)
    return np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]


def _decode_bridge(raw, coef):
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    lin = coef[1:5]
    x = np.linalg.solve(lin.T, (y-coef[0]).T).T
    for _ in range(18):
        err = _bridge_basis(x)@coef-y
        if np.max(np.abs(err), initial=0.) < 1e-10:
            break
        jac = np.broadcast_to(lin, (len(x), 4, 4)).copy()
        jac += 3*x[:, :, None]**2*coef[5:9]
        for v, (i, j) in zip(coef[9:], _PAIRS):
            jac[:, i, :] += x[:, j, None]*v
            jac[:, j, :] += x[:, i, None]*v
        step = np.linalg.solve(jac.transpose(0, 2, 1), err[:, :, None])[:, :, 0]
        x -= step/np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return x.reshape(np.shape(raw))
