import numpy as np
from scipy.linalg import solve


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_CUBIC = sorted({tuple(sorted((i, j, k)))
                 for i, j in _EDGES for k in range(4)})


def _features(z):
    pairs = np.column_stack([z[:, i] * z[:, j] for i, j in _EDGES])
    cubic = np.column_stack([z[:, i] * z[:, j] * z[:, k]
                             for i, j, k in _CUBIC])
    squares = z**2
    modulated_pairs = (pairs[:, :, None] * squares[:, None, :]).reshape(len(z), 16)
    return np.column_stack((np.ones(len(z)), z, squares, pairs, cubic,
                            modulated_pairs, squares**2))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    n = data['n_train']
    features = _features(data['z'])
    scale = np.sqrt(np.mean(features[:n]**2, axis=0))
    train = features[:n] / scale
    targets = np.column_stack((data['y'], data['traj']))
    # Fit all horizons on the same normalized basis with mild shrinkage.
    gram = train.T @ train + (1e-4 * n) * np.eye(train.shape[1])
    coefficients = solve(gram, train.T @ targets, assume_a='pos')
    prediction = (features[n:] / scale) @ coefficients
    response, trajectory = prediction[:, 0], prediction[:, 1:]
    final_state = trajectory[:, -1]
    return {'y_hat': response, 'state_pred': trajectory,
            'diagnostics_vec': np.array([response.mean(), response.std(),
                                         final_state.mean(), final_state.std()])}
