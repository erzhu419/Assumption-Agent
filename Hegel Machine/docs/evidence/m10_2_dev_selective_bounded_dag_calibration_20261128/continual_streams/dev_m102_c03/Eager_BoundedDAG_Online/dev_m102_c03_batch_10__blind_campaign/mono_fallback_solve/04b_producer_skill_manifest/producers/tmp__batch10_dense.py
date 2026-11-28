"""Infer operators from the calibration panels, then fit the observable readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    p = study['panel_a']
    u, v = np.asarray(p['u']), np.asarray(p['v'])
    def a_features(z, j):
        f, g = z[:, j], z[:, (j+1) % 4]
        return np.column_stack((np.ones(len(z)), f, g, f*g, f**3))
    designs = [a_features(u, j) for j in range(4)]
    keep = np.arange(len(u))
    for _ in range(7):
        pars = np.array([_ls(f[keep], v[keep, j]) for j, f in enumerate(designs)])
        residual = np.linalg.norm(np.column_stack([f @ pars[j] for j, f in enumerate(designs)])-v, axis=1)
        new_keep = np.flatnonzero(residual <= max(0.001, 5*np.median(residual)))
        if len(new_keep) < 12 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    # Invert the cyclic cubic map by Newton iterations, independently per row.
    z = x.copy()
    for _ in range(25):
        w = np.roll(z, -1, axis=1)
        res = pars[:,0] + pars[:,1]*z + pars[:,2]*w + pars[:,3]*z*w + pars[:,4]*z**3-x
        if np.max(np.abs(res), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z),4,4))
        j = np.arange(4)
        jac[:,j,j] = pars[:,1]+pars[:,3]*w+3*pars[:,4]*z*z
        jac[:,j,(j+1)%4] = pars[:,2]+pars[:,3]*z
        z -= np.clip(np.linalg.solve(jac, res[...,None])[...,0], -0.7, 0.7)
    return z


def _panels(study, z):
    c = study['panel_c']
    u, v1, v2 = (np.asarray(c[k]) for k in ('u', 'v1', 'v2'))
    # The sine component shares the mixing matrix with tanh after a cyclic shift.
    def first(u, p):
        a, b, c = p
        return _bias(np.tanh(a*u)+c*np.roll(np.sin(b*u), 1, axis=1))
    def second(u, p):
        return _bias(np.cos(p[0]*u))
    def proj(p, make, v):
        f = make(u, p)
        return (f @ _ls(f, v)-v).ravel()
    p1 = least_squares(proj, [1.75,2.1,0.3], args=(first,v1), bounds=([1,1.5,0],[2.5,2.8,.6])).x
    p2 = least_squares(proj, [2.4], args=(second,v2), bounds=([1.8],[3.])).x
    return np.column_stack([f(z,p) @ _ls(f(u,p),v) for f,p,v in ((first,p1,v1),(second,p2,v2))])


def _quadratic(w):
    i, j = np.triu_indices(w.shape[1])
    return _bias(np.column_stack((w,w[:,i]*w[:,j])))


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x, xt = np.asarray(tr['x']), np.asarray(study['test']['x'])
    w = _panels(study, _coordinates(study, np.vstack((x,xt))))
    f = _quadratic(w)
    fit = f[:len(x)]
    traj = np.asarray(tr['traj'])
    _, _, temporal = np.linalg.svd(traj, full_matrices=False)
    basis = temporal[:2]
    target = np.column_stack((tr['y'], traj @ basis.T))
    alpha = 0.0005
    coef = np.linalg.solve(fit.T@fit+alpha*np.eye(fit.shape[1]),fit.T@target)
    p = f[len(x):] @ coef
    y = p[:,0]
    state = p[:,1:] @ basis
    return {'y_hat':y, 'state_pred':state, 'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
