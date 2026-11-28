import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))


def panel_coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    design = np.column_stack((np.ones(len(u)), u))
    preliminary = np.linalg.lstsq(design, v, rcond=None)[0]
    error = np.linalg.norm(v - design @ preliminary, axis=1)
    keep = error < 2
    u, v = u[keep], v[keep]
    du, dv = u - u.mean(0), v - v.mean(0)
    gain = (du * dv).sum(0) / (du * du).sum(0)
    offset = (v - u * gain).mean(0)
    b = study['panel_b']
    ub, vb = b['u'], b['v']
    mat = np.linalg.lstsq(ub - ub.mean(0), vb - vb.mean(0), rcond=None)[0]
    left, values, right = np.linalg.svd(mat)
    mat = values.mean() * (left @ right)
    shift = vb.mean(0) - ub.mean(0) @ mat
    direct = (x * gain + offset - shift) @ np.linalg.inv(mat)
    inverse = ((x - offset) / gain - shift) @ np.linalg.inv(mat)
    return direct, inverse


def temporal_basis(t, traj):
    def basis(p):
        return np.column_stack((-np.expm1(-p[0] * t), np.exp(-p[2] * t) * np.sin(p[1] * t)))

    def residual(p):
        b = basis(p)
        return (b @ np.linalg.lstsq(b, traj.T, rcond=None)[0] - traj.T).ravel()

    p = least_squares(residual, [0.55, 1.13, 0.18], bounds=([0.02, 0.5, -0.3], [3, 2, 1])).x
    return basis(p)


def additive_kernel(w, length):
    d = w[:, None, :] - w[None, :, :]
    return np.exp(-d**2 / (2 * length**2)).sum(2)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    n = len(train['x'])
    f = quadratic(x)
    p = f @ f.T
    d = cdist(x, x, 'sqeuclidean')
    short = 0.1 * np.exp(-d / (2 * 1.1**2))
    long = np.exp(-d / (2 * 1.6**2))
    direct, inverse = panel_coordinates(study, x)

    def predict(k, targets, noise):
        alpha = solve(k[:n, :n] + noise * np.eye(n), targets, assume_a='pos')
        return k[n:, :n] @ alpha

    scalar = p + 0.03 * additive_kernel(direct, 0.45)
    y = 0.5 * predict(scalar + short, train['y'], 0.0003)
    y += 0.5 * predict(scalar + long, train['y'], 0.0003)
    traj = train['traj']
    basis = temporal_basis(study['eval_times'], traj)
    modes = np.linalg.lstsq(basis, traj.T, rcond=None)[0].T
    state = predict(p + long + 0.2 * additive_kernel(inverse, 1.5), modes, 0.002) @ basis.T
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
