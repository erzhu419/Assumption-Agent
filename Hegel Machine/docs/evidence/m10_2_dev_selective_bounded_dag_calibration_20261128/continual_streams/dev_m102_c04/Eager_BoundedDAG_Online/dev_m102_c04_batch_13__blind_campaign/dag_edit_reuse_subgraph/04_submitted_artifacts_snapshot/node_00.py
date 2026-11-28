import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    a = _affine(u)
    mask = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = _ls(a[mask], v[mask])
        e = np.linalg.norm(a @ w - v, axis=1)
        new = e <= max(1e-10, 4 * np.median(e))
        if new.sum() < 8 or np.array_equal(mask, new):
            break
        mask = new
    d = np.array([_ls(a[mask][:, [0, j+1]], v[mask, j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask], axis=1)) <= 1.35*np.median(e[mask])+1e-10:
        return d[0], np.diag(d[1])
    return w[0], w[1:]


def _panel_operators(panel):
    u = panel['u']
    def design(x, k, j):
        if j == 1:
            h = x / (1 + k * np.abs(x))
        else:
            h = x*x / (1 + k*x*x)
        return np.column_stack((_affine(x), h))
    def fit(j, initial):
        v = panel[f'v{j}']
        def residual(k):
            f = design(u, k[0], j)
            return (f @ _ls(f, v) - v).ravel()
        k = least_squares(residual, [initial], bounds=(0.01, 5.)).x[0]
        w = _ls(design(u, k, j), v)
        return lambda x: design(x, k, j) @ w
    return fit(1, 0.8), fit(2, 0.16)


def _features(p, r):
    # A common diagonal coefficient is represented by the dot product.
    common = np.column_stack((p, r, np.einsum('ij,ij->i', p, r)))
    offdiag = np.column_stack([p[:, i]*r[:, j] for i in range(4)
                               for j in range(4) if i != j])
    return common, offdiag


def _ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    w = np.linalg.solve(a.T @ a + penalty*np.eye(f.shape[1]), a.T @ (target-ym))
    return (query-fm) @ w + ym


CONFIG = (1, 4.0, 0.1)

def _joint(panel, q, p, target, trajectory, times):
    from itertools import combinations_with_replacement
    from scipy.linalg import svd
    u, z = panel['u'], panel['v2']
    n, d = len(target), u.shape[1]
    i,j=np.triu_indices(d)
    tri=np.array(list(combinations_with_replacement(range(d),3)))
    mode, weight, reg = CONFIG
    def design(x):
        a=_affine(x); cross=x[:,i]*x[:,j]; xx=(x*x).sum(axis=1,keepdims=True)
        if mode==0: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j]]
        if mode==1: return np.c_[a, cross, .33*np.prod(x[:,tri],axis=2)/(1+.16*xx)]
        if mode==2: return np.c_[a,cross[:,i!=j],.35*cross[:,i==j],.65*cross/(1+.3*xx)-.65*cross]
        if mode==3: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j],.50*cross/(1+.22*cross*cross)-.50*cross]
        if mode==4: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j],.5*x*x/(1+.22*x*x),.5*x/(1+.22*abs(x))]
    a,b=design(u),design(q)
    ref = _ls(a,z) if reg==0 else np.linalg.solve(a.T@a+reg*np.eye(a.shape[1]),a.T@z)
    _,_,tt=svd(trajectory,full_matrices=False)
    ts=trajectory@tt[:2].T
    yy = np.c_[target,ts]
    scl = np.maximum(yy.std(0), .07)
    ys = (yy-yy.mean(0))/scl
    def make(w):return _features(p,b@w.reshape(-1,d))
    g,gx=make(ref)
    def select(g,gx,y,cutoff):
        h=_affine(g[:n]); e=y-h@_ls(h,y); c=gx[:n]-h@_ls(h,gx[:n]); gain=np.sum((c.T@e)**2,axis=1)/np.maximum(np.sum(c*c,axis=0),1e-16); j=np.argmax(gain)
        return j if gain[j] > cutoff*np.sum(e*e) else None
    def add(g,gx,ix):return g if ix is None else np.c_[g,gx[:,ix]]
    iy=select(g,gx,ys[:,:1],.15); it=select(g,gx,ts,.15)
    def resid(w):
        g,gx=make(w); iyf=add(g,gx,iy); itf=add(g,gx,it)
        ey=_ridge(iyf[:n],ys[:,0],iyf[:n],1e-5)-ys[:,0]; et=_ridge(itf[:n],ys[:,1:],itf[:n],1e-5)-ys[:,1:]
        return np.r_[((a@w.reshape(-1,d)-z)/np.sqrt(len(a))).ravel(), weight*np.r_[ey,et.ravel()]/np.sqrt(n), np.sqrt(reg/len(a))*(w-ref.ravel())]
    w=least_squares(resid,ref.ravel(),max_nfev=45,ftol=2e-5).x
    g,gx=make(w)
    iy=select(g,gx,ys[:,:1],.15)
    yf=add(g,gx,iy)
    return g,gx,_ridge(yf[:n], target, yf[n:],1e-6)

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']; x,xt=train['x'],study['test']['x']
    offset,matrix=_calibration(study['panel_a'])
    q=np.linalg.solve(matrix.T,(np.concatenate((x,xt))-offset).T).T
    p,_=_panel_operators(study['panel_c'])
    common, offdiag, y = _joint(study['panel_c'],q,p(q),train['y'],train['traj'],None)
    return {'common':common,'offdiag':offdiag,'y_hat':y}
