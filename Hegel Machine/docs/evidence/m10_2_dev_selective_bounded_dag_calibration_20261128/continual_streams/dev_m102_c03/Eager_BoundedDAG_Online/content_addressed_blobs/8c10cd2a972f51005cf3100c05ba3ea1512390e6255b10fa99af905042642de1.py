import numpy as np
from scipy.optimize import minimize_scalar


def _pairs(z):
    return z * np.roll(z, -1, axis=1)


def _ols(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(p, x):
    u, v = p['u'], p['v']
    h = _one(u)
    e = np.linalg.norm(v-h@_ols(h,v), axis=1)
    keep = e < max(1.0, 4*np.median(e))
    u, v = u[keep], v[keep]
    def design(z, i):
        a, b = z[:, i], z[:, (i+1)%4]
        return np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
    offset, diag, cross, bilinear, cubic = np.array([
        _ols(design(u, i), v[:, i]) for i in range(4)
    ]).T
    z = (x-offset)/diag
    indices = np.arange(4)
    for _ in range(30):
        neighbor = np.roll(z, -1, axis=1)
        residual = offset + diag*z + cross*neighbor + bilinear*z*neighbor + cubic*z**3 - x
        if np.max(np.abs(residual), initial=0) < 1e-11:
            break
        jac = np.zeros((len(z), 4, 4))
        jac[:, indices, indices] = diag + bilinear*neighbor + 3*cubic*z*z
        jac[:, indices, (indices+1)%4] = cross + bilinear*z
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= step / np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return z


def run_node(inputs: dict) -> dict:
    study, sparse = inputs['__input__'], inputs['node_00']
    tr = study['train']
    z = _calibrate(study['panel_a'], np.vstack((tr['x'], study['test']['x'])))
    p=study['panel_c']; u=p['u']
    hp=_one(_pairs(u)); bp=_ols(hp,p['v1'])
    def loss(w):
        h=_one(np.cos(w*u)); return np.mean((h@_ols(h,p['v2'])-p['v2'])**2)
    w=minimize_scalar(loss,bounds=(1.2,3.8),method='bounded').x
    hc=_one(np.cos(w*u)); bc=_ols(hc,p['v2'])
    P=_one(_pairs(z))@bp; C=_one(np.cos(w*z))@bc
    H=_one(np.column_stack((P,C,P*C,(P[:,:,None]*C[:,None,:]).reshape(len(z),16)[:,~np.eye(4,dtype=bool).ravel()])))
    n=len(tr['x']); Y=np.column_stack((tr['y'],tr['traj']))
    scale=np.maximum(H[:n].std(0),1e-8); scale[0]=1
    F=H/scale; reg=np.r_[0,np.full(8,.002),np.full(4,.002),np.full(12,.08)]
    pred=F[n:]@np.linalg.solve(F[:n].T@F[:n]+np.diag(reg),F[:n].T@Y)
    return {'panel_pred': np.column_stack((2*pred[:,0]-sparse['y_hat'],(pred[:,1:]-.35*sparse['state_pred'])/.65))}
