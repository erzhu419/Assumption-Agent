import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _panel_coordinates(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])

    def design(a, scale):
        return np.column_stack((np.ones(len(a)), a, a**3 / (1.0 + scale * a**2)))

    def fit(scale):
        d = design(u, scale)
        return d, np.linalg.lstsq(d, v, rcond=None)[0]

    def residual(scale):
        d, coef = fit(scale)
        return (d @ coef - v).ravel()

    scale = least_squares(residual, [0.75], bounds=(0.015, 10.0)).x
    _, coef = fit(scale)
    return design(x, scale) @ coef


def _scalar(b, bt, y):
    n = len(b)
    coords = np.vstack((b, bt))
    trend = np.column_stack((np.ones(len(coords)), coords))
    train = trend[:n]
    # Use the fitted linear score as the argument of bounded link corrections.
    beta = np.linalg.lstsq(train, y, rcond=None)[0]
    h = (trend @ beta - y.mean()) / y.std()
    links = (
        np.column_stack((np.tanh(h), np.exp(-h*h), h*h/(1+h*h))),
        np.column_stack((np.sin(h), np.cos(h), np.sin(2*h), np.cos(2*h))),
    )
    pair = (coords[:, 1] * coords[:, 3])[:, None]
    quadratic = np.column_stack(
        [coords[:, i] * coords[:, j] for i in range(4) for j in range(i+1, 4)]
        + [coords**2]
    )
    scaled = coords / (np.sqrt(b.shape[1]) * b.std(axis=0))
    base = np.exp(-0.5 * cdist(scaled, scaled[:n], 'sqeuclidean'))
    rhs = np.column_stack((y, train))
    predictions = []
    for link in links:
        for interaction in (pair, quadratic):
            features = np.column_stack((link, interaction))
            features /= np.maximum(features[:n].std(axis=0), 0.01)
            kernel = base + 0.1 * features @ features[:n].T
            for penalty in (0.01, 0.03, 0.08):
                p = solve(kernel[:n] + penalty * np.eye(n), rhs, assume_a='pos')
                beta = np.linalg.solve(train.T @ p[:, 1:], train.T @ p[:, 0])
                predictions.append(trend[n:] @ beta + kernel[n:] @ (p[:, 0] - p[:, 1:] @ beta))
    return np.mean(predictions, axis=0)


def run_node(inputs: dict) -> dict:
    study, calibrated = inputs['__input__'], inputs['node_00']
    x, xt = calibrated['x_tr'], calibrated['x_te']
    b, bt = np.split(_panel_coordinates(study['panel_b'], np.vstack((x, xt))), [len(x)])
    y_hat = _scalar(b, bt, np.asarray(study['train']['y']))
    d = np.column_stack((np.ones(len(x)), x, b))
    dt = np.column_stack((np.ones(len(xt)), xt, bt))
    state_pred = dt @ np.linalg.lstsq(d, study['train']['traj'], rcond=None)[0]
    end = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
