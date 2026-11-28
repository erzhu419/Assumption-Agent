"""Calibrate the observation and panel operators, then fit the joint readout."""
import numpy as np
from scipy.optimize import least_squares


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    keep = np.linalg.norm(v-u, axis=1) < 3
    u, v = u[keep], v[keep]
    j = np.arange(4)
    def design(z, p):
        return np.stack((np.ones_like(z), z, np.tanh(p[0]*z), np.sin(p[1]*np.roll(z, 2, axis=1))), axis=2)
    def coeff(f):
        return np.array([_ls(f[:, k], v[:, k]) for k in j])
    def residual(p):
        f = design(u, p)
        return (np.einsum('njk,jk->nj', f, coeff(f))-v).ravel()
    p = least_squares(residual, [1.8, 2.4], bounds=([.4, 1.4], [3.5, 3.5])).x
    b = coeff(design(u,p))
    z = x.copy()
    for _ in range(40):
        f = design(z, p)
        r = np.einsum('njk,jk->nj', f, b)-x
        if np.max(np.abs(r), initial=0) < 1e-11:
            break
        jac = np.zeros((len(z),4,4))
        jac[:, j, j] = b[:,1]+p[0]*b[:,2]*(1-f[:,:,2]**2)
        jac[:, j, (j+2)%4] = p[1]*b[:,3]*np.cos(p[1]*np.roll(z,2,axis=1))
        z -= np.clip(np.linalg.solve(jac,r[...,None])[...,0], -.5,.5)
    return z


def _first(u, p):
    a,b,c = p
    return _bias(np.tanh(a*u)+c*np.roll(np.sin(b*u),1,axis=1))


def _second(u,p):
    a,b,c = p
    return _bias(np.tanh(a*u)+b*u*np.roll(u,1,axis=1)+c*u)


def _panel(study,z):
    c = study['panel_c']; u = c['u']; v = c['v1']
    def residual(p):
        a = _first(u,p)
        return (a@_ls(a,v)-v).ravel()
    p = least_squares(residual,[1.75,2.1,.3],bounds=([1,1.5,0],[2.5,2.8,.6])).x
    a = _first(z,p)@_ls(_first(u,p),v)
    v2 = c['v2']
    def residual2(p):
        h = _second(u,p)
        return (h@_ls(h,v2)-v2).ravel()
    p2 = least_squares(residual2,[1.2,.45,.05],bounds=([.3,-3,-1],[3,3,2])).x
    b = _second(z,p2)@_ls(_second(u,p2),v2)
    return a,b


def _fit(study):
    t = study['train']; n = len(t['x'])
    z = _coordinates(study, np.vstack((t['x'],study['test']['x'])))
    a,b = _panel(study,z)
    f = _bias(np.column_stack((a,b,np.sum(a*b,axis=1),a[:,0]*b[:,1])))
    y = f[n:,:-1]@_ls(f[:n,:-1],t['y'])
    basis = np.linalg.svd(t['traj'],full_matrices=False)[2][:2]
    traj = f[n:]@_ls(f[:n],t['traj']@basis.T)@basis
    return y,traj


def run_node(inputs: dict) -> dict:
    y, traj = _fit(inputs['__input__']['study'] if 'study' in inputs['__input__'] else inputs['__input__'])
    b = inputs['node_00']
    return {'panel_pred': np.column_stack((2*y-b['y_hat'],(traj-.35*b['state_pred'])/.65))}
