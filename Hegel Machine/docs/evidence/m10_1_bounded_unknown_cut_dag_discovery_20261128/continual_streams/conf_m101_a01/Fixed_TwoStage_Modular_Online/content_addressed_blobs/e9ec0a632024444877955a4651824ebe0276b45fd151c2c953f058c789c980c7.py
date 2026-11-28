import numpy as np


def _standardize(x, n):
    return (x - x[:n].mean(axis=0)) / np.maximum(x[:n].std(axis=0), 1e-12)


def _scalar(z, y):
    n = len(y)
    x = _standardize(z['x'], n)
    w = _standardize(z['i2b'], n)
    trend = (1 + x @ x.T / x.shape[1]) ** 2
    d = ((w[:, None, :] - w[None, :, :]) / 1.5) ** 2
    kernel = trend + np.exp(-d.sum(axis=2) / 2) + np.exp(-d / 2).mean(axis=2) ** 2
    m = y.mean()
    weights = np.linalg.solve(kernel[:n, :n] + 0.008 * np.eye(n), y - m)
    return m + kernel[n:, :n] @ weights


def _trajectory(z, targets):
    n = len(targets)
    x = z['ib']
    cols = [x]
    cols.extend(x[:, i:i+1] * x[:, j:j+1] for i in range(4) for j in range(i, 4))
    cols.extend(f(x) for f in (np.sin, np.cos, np.tanh, np.arctan))
    cols.append(x / (1 + x*x))
    a = _standardize(np.column_stack(cols), n)
    targets_mean = targets.mean(axis=0)
    u, s, vh = np.linalg.svd(a[:n], full_matrices=False)
    return targets_mean + ((a[n:] @ vh.T) * (s / (s*s + 0.01))) @ (u.T @ (targets - targets_mean))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    z = data['z']
    y = _scalar(z, data['y'])
    state = _trajectory(z, data['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
