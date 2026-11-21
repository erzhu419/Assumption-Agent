import math
import numpy as np
from scipy.optimize import least_squares


def _hill(x, vmax, k):
    q = (max(float(x), 0.0) / 1.5) ** 2
    return vmax * q / (k + q)


def fit(train_rows):
    rows = [r for r in train_rows if 'y' in r]
    if not rows:
        return {'params': np.array([8.0, 1.0, 0.3, 0.0]), 'feature': None, 'cut': 0.0, 'direction': 1}
    y = np.array([float(r['y']) for r in rows])
    # Search simple threshold indicators over each independently varied z coordinate.
    best = None
    for j in range(1, 5):
        vals = np.array([float(r.get('z_%d' % j, 0.0)) for r in rows])
        cuts = np.unique((np.sort(vals)[:-1] + np.sort(vals)[1:]) / 2.0)
        for cut in cuts:
            for direction in (1, -1):
                ind = (vals >= cut).astype(float) if direction == 1 else (vals < cut).astype(float)
                def pred(p):
                    vmax, k, c, a = p
                    return np.array([_hill(r.get('x_ind', 0), vmax, k) - c * float(r.get('u_ctrl', 0)) for r in rows]) + a * ind
                opt = least_squares(lambda p: pred(p) - y, [8.0, 1.0, 0.3, 0.5], bounds=([0.01, 0.001, -5.0, -10.0], [30.0, 30.0, 5.0, 10.0]))
                loss = float(np.mean((pred(opt.x) - y) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, opt.x, j, float(cut), direction)
    if best is None:
        return {'params': np.array([8.0, 1.0, 0.3, 0.0]), 'feature': None, 'cut': 0.0, 'direction': 1}
    return {'params': best[1], 'feature': best[2], 'cut': best[3], 'direction': best[4]}


def predict(row):
    # Call fit(train_rows) first and attach its result as row['_model'] when using
    # a stateless evaluator; otherwise the default is the historical Hill curve.
    model = row.get('_model')
    if model is None:
        return float(_hill(row.get('x_ind', 0), 8.0, 1.0) - 0.3 * float(row.get('u_ctrl', 0)))
    vmax, k, c, a = model['params']
    value = _hill(row.get('x_ind', 0), vmax, k) - c * float(row.get('u_ctrl', 0))
    j = model.get('feature')
    if j is not None:
        z = float(row.get('z_%d' % j, 0.0))
        hit = z >= model['cut'] if model['direction'] == 1 else z < model['cut']
        value += a * float(hit)
    return float(value)
