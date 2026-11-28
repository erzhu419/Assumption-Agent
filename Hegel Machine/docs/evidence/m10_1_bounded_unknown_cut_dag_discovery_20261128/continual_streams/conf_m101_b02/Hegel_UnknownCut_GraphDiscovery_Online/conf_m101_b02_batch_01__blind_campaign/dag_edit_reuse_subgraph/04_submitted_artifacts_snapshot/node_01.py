import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _features(s, prev):
    z, b = prev['z'], prev['b']
    panel = s['panel_c']
    u = panel['u']
    def c1(x, k):
        return _design(np.column_stack((x, x/(1+k*abs(x)))))
    def c2(x, k, c):
        return _design(x+c*(np.logaddexp(k*x, -k*x)-np.log(2)))
    out, errors = [z, b], []
    for name, basis, init, bounds in [('v1', c1, [.83], ([.1], [4.])), ('v2', c2, [1.1, .91], ([.3, .1], [3., 4.]))]:
        v = panel[name]
        def residual(q):
            m = basis(u, *q)
            return (m @ _fit(m, v)-v).ravel()
        q = least_squares(residual, init, bounds=bounds).x
        m = basis(u, *q)
        p = _fit(m, v)
        out.extend([basis(z, *q) @ p, basis(b, *q) @ p])
        errors.append(np.sqrt(np.mean((m @ p-v)**2)))
    return out, errors


def _time(t, traj):
    def basis(q):
        return -np.expm1(-np.outer(t, q))
    def residual(q):
        h = basis(q)
        return (h @ _fit(h, traj.T)-traj.T).ravel()
    return basis(least_squares(residual, [.41, 1.81], bounds=([.03, .7], [.7, 6.])).x)


def _kernel(groups, train):
    # Keep the latent coordinates, the B output, and the two B-to-C routes.
    grams = []
    for x in (groups[0], groups[1], groups[3], groups[5]):
        x = (x-x[train].mean(0))/np.maximum(x[train].std(0), 1e-9)
        grams.append(x @ x.T / x.shape[1])
    route = sum(grams[1:])
    p = (2*grams[0]+route)/4
    return 1+p+grams[0]*route/4+.3*p**2


def run_node(inputs: dict) -> dict:
    s, prev = inputs['__input__'], inputs['node_00']
    groups, _ = _features(s, prev)
    train = s['train']
    n = len(train['y'])
    h = _time(s['eval_times'], train['traj'])
    y = np.column_stack((train['y'], _fit(h, train['traj'].T).T))
    K = _kernel(groups, np.arange(n))
    pred = K[n:, :n] @ solve(K[:n, :n]+.003*np.eye(n), y, assume_a='pos')
    y_hat = pred[:, 0]
    state_pred = pred[:, 1:] @ h.T
    final = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])}
