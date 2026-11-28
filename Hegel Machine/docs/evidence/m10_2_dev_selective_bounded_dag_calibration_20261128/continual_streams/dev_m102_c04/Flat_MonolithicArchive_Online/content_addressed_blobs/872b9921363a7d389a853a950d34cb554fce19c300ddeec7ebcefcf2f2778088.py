import numpy as np
from scipy.optimize import least_squares, minimize_scalar

_PAIRS = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(*cols):
    return np.column_stack((np.ones(len(cols[0])), *cols))


def _calibrate(panel, observed):
    u, v = panel['u'], panel['v']
    d = np.max(np.abs(u-v), axis=1)
    keep = d < max(0.5, 4*np.median(d))
    coef = np.column_stack([_ls(_aff(u[keep, j]), v[keep, j]) for j in range(u.shape[1])])
    return (observed-coef[0])/coef[1]


def _operators(panel, z):
    u = panel['u']
    def first(x, k):
        return _aff(x, x/(1+k*np.abs(x)))
    def second(x, k):
        return _aff(x, np.arctan(k*x), *(x[:,i]*x[:,j] for i,j in _PAIRS if (i,j) not in ((0,2),(1,3))))
    outputs = []
    for key, basis, interval in [('v1', first, (.3,1.5)), ('v2', second, (.5,1.7))]:
        v = panel[key]
        def score(k):
            a = basis(u,k)
            return np.sum((a@_ls(a,v)-v)**2)
        k = minimize_scalar(score, bounds=interval, method='bounded', options={'xatol': 1e-9}).x
        outputs.append(basis(z,k)@_ls(basis(u,k),v))
    return outputs


def _predict_state(times, traj, common, r, cross):
    """Fit both temporal rates jointly, profiling out spatial coefficients."""
    n, nt = traj.shape
    target = traj.ravel()
    best_score = np.inf
    for j in range(cross.shape[1]):
        fast = np.column_stack((common, cross[:, j]))
        slow = _aff(r, cross[:, j])

        def design(rates, rows):
            basis = -np.expm1(-times[:, None] * rates)
            return np.concatenate((fast[rows, None, :] * basis[None, :, 0, None],
                                   slow[rows, None, :] * basis[None, :, 1, None]),
                                  axis=2).reshape(-1, fast.shape[1] + slow.shape[1])

        def residual(rates):
            a = design(rates, slice(None, n))
            return a @ _ls(a, target) - target

        fit = least_squares(residual, [1.8, .43], bounds=([.8, .03], [8., .79]))
        if fit.cost < best_score:
            best_score = fit.cost
            a = design(fit.x, slice(None, n))
            pred = (design(fit.x, slice(n, None)) @ _ls(a, target)).reshape(len(r)-n, nt)
    return pred


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'],study['test']['x']))
    p,r = _operators(study['panel_c'],_calibrate(study['panel_a'],x))
    dot = np.sum(p*r,axis=1)
    scalar = _aff(p,r,dot)
    y = scalar[n:]@_ls(scalar[:n],train['y'])
    cross = (p[:,:,None]*r[:,None,:]).reshape(len(p),-1)
    state = _predict_state(study['eval_times'], train['traj'], scalar, r, cross)
    last = state[:,-1]
    return {'y_hat':y, 'state_pred':state,
            'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}
