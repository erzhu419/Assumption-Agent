import numpy as np
from scipy.optimize import least_squares

def fit_a(panel):
    u, v = panel['u'], panel['v']
    n = np.roll(u, -1, axis=1)
    D = np.stack((np.ones_like(u), u, n*(u-1), u**3), axis=-1)
    w = np.ones(len(u))
    for _ in range(24):
        c = np.array([np.linalg.lstsq(D[:,j]*w[:,None], v[:,j]*w,rcond=None)[0] for j in range(4)]).T
        e = np.linalg.norm(np.einsum('njk,kj->nj',D,c)-v, axis=1)
        w = 1/np.sqrt(1+(e/max(0.001, 2*np.median(e)))**2)
    keep = np.argsort(e)[:-3]
    return np.array([np.linalg.lstsq(D[keep,j], v[keep,j],rcond=None)[0] for j in range(4)]).T

def inv_a(x,c):
    b,d,q,g = c
    A=np.diag(d); A[np.arange(4),(np.arange(4)+1)%4]=-q
    u=np.linalg.solve(A,(x-b).T).T
    for i, row in enumerate(x):
        u[i]=least_squares(lambda y:b+d*y+q*(y-1)*np.roll(y,-1)+g*y**3-row,u[i],gtol=1e-12,ftol=1e-12,xtol=1e-12).x
    e=b+d*u+q*(u-1)*np.roll(u,-1,axis=1)+g*u**3-x
    return u


def _basis_b(x, p):
    a, f = p
    w = x + a*np.sin(f*x)
    return np.column_stack((np.ones(len(x)), w, x*np.roll(x,-1,axis=1)))

def _fit_b(panel):
    u, v = panel['u'], panel['v']
    def fit(p):
        d = _basis_b(u,p)
        return d, np.linalg.lstsq(d,v,rcond=None)[0]
    def residual(p):
        d, c = fit(p)
        return (d@c-v).ravel()
    p = least_squares(residual,[-.4,2.8],bounds=([-2,1],[1,5])).x
    return p, fit(p)[1]

def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = fit_a(study['panel_a'])
    p, b = _fit_b(study['panel_b'])
    return {key: _basis_b(inv_a(study[split]['x'],a),p)@b
            for key, split in [('z_tr','train'),('z_te','test')]}


def _coordinates(study, mode, penalty):
    x = np.concatenate((study['train']['x'], study['test']['x']))
    y = np.asarray(study['train']['y']).ravel()
    traj = np.asarray(study['train']['traj'])
    n = len(y)
    a = fit_a(study['panel_a'])
    t = inv_a(x, a)
    p, b = _fit_b(study['panel_b'])
    z0 = _basis_b(t, p) @ b
    if mode == 0:
        return z0
    _, _, vt = np.linalg.svd(traj-traj.mean(0), full_matrices=False)
    state = (traj-traj.mean(0)) @ vt[0] / np.sqrt(traj.shape[1])
    sy, ss = np.std(y), np.std(state)
    ix = np.arange(4)
    J = np.zeros((len(t),4,4))
    J[:,ix,ix] = a[1]+a[2]*np.roll(t,-1,axis=1)+3*a[3]*t*t
    J[:,ix,(ix+1)%4] = a[2]*(t-1)
    J = np.linalg.inv(J)
    def make_f(z):
        return np.column_stack((np.ones(len(z)),z,z[:,0]*z[:,2],z[:,1]*z[:,3]))
    def design(z,d):
        F = make_f(z)
        q = F[:,:5] @ d
        return np.column_stack((F, np.column_stack((q,F[:,5:]))/(1+q*q)[:,None]))
    def error(z,d):
        F = design(z,d)
        G = F[:,:5]
        return np.r_[(F@np.linalg.lstsq(F,y,rcond=None)[0]-y)/sy,
                     (G@np.linalg.lstsq(G,state,rcond=None)[0]-state)/ss]
    if mode >= 4:
        pairs = [(i,j) for i in range(4) for j in range(i,4)]
        w = t+p[0]*np.sin(p[1]*t)
        features = [np.sin(p[1]*t), t*np.cos(p[1]*t), np.cos(p[1]*t),
                    np.column_stack([t[:,i]*t[:,j] for i,j in pairs])]
        if mode == 5:
            features.extend([t-np.sin(p[1]*t)/p[1], w*np.roll(w,-1,axis=1), t**3])
        H = np.column_stack(features)
        H = (H-H[:n].mean(0))/np.maximum(H[:n].std(0),.05)
        # Restrict additive corrections to the two directions observed by telemetry.
        base = make_f(z0[:n])
        dirs = np.stack((np.linalg.lstsq(base,y,rcond=None)[0][1:5],
                         np.linalg.lstsq(base[:,:5],state,rcond=None)[0][1:5]))
        dirs = np.linalg.qr(dirs.T)[0].T
        k = H.shape[1]*2
        def coords(v, sl):
            return z0[sl]+(H[sl]@v.reshape(-1,2))@dirs
    else:
        k = {1:8,2:20,3:30}[mode]
        def coords(v, sl):
            dx = .18*v[:4]+.12*x[sl]*v[4:8]
            h = t[sl]+np.einsum('nij,nj->ni',J[sl],dx)
            if mode == 1:
                return _basis_b(h,p)@b
            freq = p[1]+.6*v[8:12]
            amp = p[0]+.17*v[12:16]
            phase = .4*v[16:20]
            z = np.column_stack((np.ones(len(h)),h+amp*np.sin(h*freq+phase),h*np.roll(h,-1,axis=1)))@b
            if mode==3:
                pairs = [(i,j) for i in range(4) for j in range(i,4)]
                F = make_f(z0[:n]); dr = np.linalg.lstsq(F[:,:5],state,rcond=None)[0][1:5]; dr /= np.linalg.norm(dr)
                z += (.12*np.column_stack([h[:,i]*h[:,j] for i,j in pairs])@v[20:])[:,None]*dr
            return z
    sl = slice(None,n)
    d0 = np.linalg.lstsq(make_f(z0[sl]),y,rcond=None)[0][:5]
    d0 = d0 / max(np.std(make_f(z0[sl])[:,:5]@d0), 1e-8)
    def objective(v):
        return np.r_[error(coords(v[5:],sl),v[:5]), penalty*v[5:], .016*v[:5]]
    starts = [d0*.65,d0*1.65,np.array([0.,1.,-.5,1.,-.5])]
    # First profile the decoder at the calibrated coordinates; use that solution to seed the joint fit.
    starts = [least_squares(lambda d:error(z0[sl],d), d, max_nfev=38).x for d in starts]
    starts.sort(key=lambda d:np.linalg.norm(error(z0[sl],d)))
    fits = [least_squares(objective,np.r_[d,np.zeros(k)],max_nfev=95,ftol=2e-7) for d in starts[:2]]
    best = min(fits,key=lambda f:f.cost)
    return coords(best.x[5:],slice(None))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z = _coordinates(study, MODEL, PENALTY)
    n = len(study['train']['x'])
    return {'z_tr':z[:n], 'z_te':z[n:]}
