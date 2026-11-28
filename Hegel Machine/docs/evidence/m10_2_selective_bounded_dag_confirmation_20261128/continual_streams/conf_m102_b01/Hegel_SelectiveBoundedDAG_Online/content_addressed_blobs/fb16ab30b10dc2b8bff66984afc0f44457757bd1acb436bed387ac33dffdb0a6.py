from itertools import combinations_with_replacement
import numpy as np


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _trajectory_terms(x, y):
    n = len(y)
    terms = list(range(x.shape[1]))

    def criterion(cols):
        a = x[:, cols]
        residual = y - a @ _fit(a, y)
        return n * np.log(max(np.mean(residual**2), 1e-30)) + len(cols) * np.log(n)

    while len(terms) > 1:
        value, drop = min((criterion([j for j in terms if j != i]), i) for i in terms)
        if value >= criterion(terms):
            break
        terms.remove(drop)
    return terms


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    w = np.asarray(inputs['node_00']['w'], dtype=float)
    y = np.asarray(train['y'], dtype=float)
    trajectory = np.asarray(train['traj'], dtype=float)
    pairs = list(combinations_with_replacement(range(w.shape[1]), 2))
    products = [w[:, i] * w[:, j] for i, j in pairs]
    features = np.column_stack([np.ones(len(w)), w, *products])
    n = len(y)
    a, b = features[:n], features[n:]
    terms = _trajectory_terms(a, trajectory)
    # The scalar head is quadratic in the active bilinear coordinates,
    # not a general fourth-degree polynomial in the original coordinates.
    offset = 1 + w.shape[1]
    active = [j for j in terms if j >= offset and
              pairs[j - offset][0] != pairs[j - offset][1]]
    k = len(active)
    if k and 1 + k + k * (k + 1) // 2 + 5 <= n:
        scalar = np.column_stack([np.ones(len(w)), features[:, active],
                                  *[features[:, i] * features[:, j]
                                    for i, j in combinations_with_replacement(active, 2)]])
        y_hat = scalar[n:] @ _fit(scalar[:n], y)
    else:
        y_hat = b @ _fit(a, y)
    state_pred = b[:, terms] @ _fit(a[:, terms], trajectory)
    tail = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), tail.mean(), tail.std()])}
