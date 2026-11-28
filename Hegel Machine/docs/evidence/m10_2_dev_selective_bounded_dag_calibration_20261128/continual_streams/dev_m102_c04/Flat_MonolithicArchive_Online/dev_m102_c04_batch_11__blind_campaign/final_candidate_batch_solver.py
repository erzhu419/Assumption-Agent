"""Calibrated dual-operator predictor with a two-rate trajectory basis."""
import numpy as np
from scipy.optimize import least_squares


def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    mask = np.max(np.abs(u-v), axis=1) < 3
    u, v = u[mask], v[mask]
    j = np.arange(4)
    eye = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    base = (eye, eye*u[:, :, None])

    def profile(rate):
        a = np.concatenate((*base, np.tanh(rate[0]*u)[:, :, None],
                            np.sin(rate[1]*np.roll(u, 2, axis=1))[:, :, None]), axis=2).reshape(-1, 10)
        w = _fit(a, v.ravel())
        return a @ w-v.ravel(), w

    k = least_squares(lambda p: profile(p)[0], [1.8, 2.4], bounds=([.5, 1.5], [4., 3.3])).x
    _, w = profile(k)
    offset, gain, a, b = w[:4], w[4:8], w[8], w[9]
    z = (observed-offset)/gain
    # Fixed-point warm start; refine against the original equations below.
    for _ in range(55):
        update = (observed-offset-a*np.tanh(k[0]*z)-b*np.sin(k[1]*np.roll(z, 2, axis=1)))/gain
        if np.max(np.abs(update-z), initial=0.) < 1e-12:
            break
        z = update
    # Newton refinement (the self term is monotone and dominates the coupling).
    for _ in range(12):
        r = np.roll(z, 2, axis=1)
        h = np.tanh(k[0]*z)
        err = offset+gain*z+a*h+b*np.sin(k[1]*r)-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z),4,4))
        jac[:,j,j] = gain+a*k[0]*(1-h*h)
        jac[:,j,(j+2)%4] = b*k[1]*np.cos(k[1]*r)
        z -= np.linalg.solve(jac, err[...,None])[...,0]
    return z


def _operators(panel, z):
    u = np.asarray(panel['u'], dtype=float)
    out = []
    for key, fn, k0 in [('v1', lambda x,k: x/(1+k*np.abs(x)), .83),
                        ('v2', lambda x,k: np.log1p(k*x*x), .36)]:
        v = np.asarray(panel[key], dtype=float)
        def residual(k):
            a = _design(u, fn(u,k))
            return (a @ _fit(a,v)-v).ravel()
        k = least_squares(residual, [k0], bounds=(.01,4.)).x
        out.append(_design(z,fn(z,k)) @ _fit(_design(u,fn(u,k)),v))
    return out


def _time_basis(t, measured):
    def design(k):
        return -np.expm1(-t[:,None]*k)
    def residual(k):
        a = design(k)
        return (a @ _fit(a,measured.T)-measured.T).ravel()
    rates = least_squares(residual,[1.8,.42],bounds=([.8,.03],[8.,.79])).x
    return np.linalg.qr(design(rates), mode='reduced')[0]


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'],study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    p, r = _operators(study['panel_c'], z)
    dot = np.sum(p*r,axis=1)
    base = _design(p,r,dot)
    cross = np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i != j])
    y = base[n:] @ _fit(base[:n],np.asarray(train['y']))
    traj = np.asarray(train['traj'])
    basis = _time_basis(np.asarray(study['eval_times']),traj)
    amplitudes = traj @ basis
    # Screen off-diagonal terms using only the training trajectories, then refit.
    dense = np.column_stack((base[:n],cross[:n]))
    interactions = _fit(dense,amplitudes)[-cross.shape[1]:]
    selected = np.argmax(np.linalg.norm(interactions,axis=1))
    space = np.column_stack((base,cross[:,selected]))
    state = (space[n:] @ _fit(space[:n],amplitudes)) @ basis.T
    last = state[:,-1]
    return {'y_hat':y,'state_pred':state,
            'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}
