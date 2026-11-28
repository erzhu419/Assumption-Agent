import numpy as np
from itertools import combinations_with_replacement


def calibrate(p, x):
    u, v = p['u'], p['v']
    delta = np.max(np.abs(v - np.median(v, axis=0)), axis=1)
    good = delta < 2.8*np.median(delta)
    u, v = u[good], v[good]
    c = []
    for i in range(4):
        q, w = u[:, i], u[:, (i+1)%4]
        c.append(np.linalg.lstsq(np.c_[np.ones(len(u)), q, w*(1+q), q**3], v[:, i], rcond=None)[0])
    b, a, g, h = np.array(c).T
    z = (x-b)/a
    i = np.arange(4)
    j = (i+1)%4
    for _ in range(25):
        w = z[:, j]
        r = b+a*z+g*w*(1+z)+h*z**3-x
        jac = np.zeros((len(z), 4, 4))
        jac[:, i, i] = a+g*w+3*h*z*z
        jac[:, i, j] = g*(1+z)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step/np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return z


def poly(x, degree):
    return np.column_stack([np.prod(x[:, j], axis=1) for d in range(degree+1) for j in combinations_with_replacement(range(x.shape[1]), d)])



def _saturate(x, g, kind):
    if kind == 0:
        return np.sin(g*x)
    if kind == 1:
        return np.tanh(g*x)
    return x/(1+(g*x)**2)


def _features(z, g, kind, form):
    b = _saturate(z, g, kind)
    cross = (z[:, :, None]*b[:, None, :]).reshape(len(z), -1)
    a = np.c_[poly(z, 2), b, cross]
    if form == 1:
        a = np.c_[a, poly(b, 2)[:, 5:]]
    elif form == 2:
        a = np.c_[a, b*b]
    elif form == 3:
        a = np.c_[a, z**3]
    elif form == 4:
        a = np.c_[a, np.column_stack([_saturate(z[:, i]+sgn*z[:, j],g,kind) for i in range(4) for j in range(i+1,4) for sgn in (-1,1)])]
    elif form == 5:
        a = np.c_[poly(z,2),poly(z-.36*b,2)[:,5:],b]
    elif form == 6:
        a = np.c_[a, (z[:, :, None]**2*b[:, None, :]).reshape(len(z),-1), poly(b, 2)[:,5:]]
    return a[:,1:]


def _regression(a, b, y, lam):
    mu = a.mean(0)
    scale = np.maximum(a.std(0), 1e-8)
    x, q = (a-mu)/scale, (b-mu)/scale
    m = y.mean(0)
    if lam:
        if x.shape[1] > len(x):
            beta = x.T @ np.linalg.solve(x@x.T+lam*np.eye(len(x)), y-m)
        else:
            beta = np.linalg.solve(x.T@x+lam*np.eye(x.shape[1]), x.T@(y-m))
    else:
        beta = np.linalg.lstsq(x,y-m,rcond=1e-9)[0]
    return q@beta+m


def _panel_shapes(s, z):
    from scipy.optimize import minimize_scalar
    f = []
    for k, j, kind, lo, hi in [('panel_b','v',0,2.4,3.3), ('panel_c','v1',1,1.5,2.8)]:
        u,v = s[k]['u'],s[k][j]
        def des(x,g):
            return np.c_[np.ones(len(x)),x,_saturate(x,g,kind)]
        def loss(g):
            a=des(u,g); q=np.linalg.qr(a)[0]
            return np.mean(((v-q@(q.T@v))/np.maximum(1e-4,1-(q*q).sum(1))[:,None])**2)
        g = minimize_scalar(loss,bounds=(lo,hi),method='bounded').x
        a=des(u,g); f.append(des(z,g)@np.linalg.lstsq(a,v,rcond=None)[0])
    return f


def _library(s,z,target):
    pars = ((0,(2.5,2.7,2.85,3.,3.2,3.35,3.5,3.65,3.8,4.)), (1,(1.15,1.35,1.55,1.75,1.95,2.2,2.5,2.8)), (2,(.65,.8,.95,1.1,1.25,1.45,1.65)))
    for k, gs in pars:
        for g in gs:
            for form in (0,1,2,3,4,5,6):
                a=_features(z,g,k,form)
                for lam in (.0002,.005,.035,.15):
                    yield (k,g,form,lam),a,None,lam
                if target==0 and form in (0,2,5):
                    r2=(z*z).sum(1)
                    for j, coord in enumerate([r2,*list(z.T**2),*[(z[:,i]+sgn*z[:,j])**2 for i in range(4) for j in range(i+1,4) for sgn in (-1,1)]]):
                        for h in (.25,.75):
                            if (k==0 and g in (3.2,3.5,3.8) or k==1 and g in (1.55,1.95) or k==2 and g in (.95,1.25)):
                                d=1+h*coord
                                yield ('rational',k,g,form,j,h),np.c_[1/d,a/d[:,None]],None,.01
    v,w=_panel_shapes(s,z)
    for k,x in enumerate((v,w,np.c_[v,w], np.c_[z,v], np.c_[z,w], np.c_[z,v,w])):
        a=poly(x,2)[:,1:]
        for lam in (.001,.01,.06,.3,1.):
            yield ('panel',k,lam),a,None,lam
    if target==0:
        # Pairwise Fourier terms: both sums and differences via product identities.
        for g in (1.2,1.6,2,2.4,2.8,3.2,3.6,4):
            b=np.sin(g*z); c=np.cos(g*z)
            for mode,f in enumerate((np.c_[z,b,c],np.c_[z,b],np.c_[b,c])):
                a=poly(f,2)[:,1:]
                for lam in (.002,.025,.10,.4):
                    yield ('fourier',g,mode,lam),a,None,lam


def _predict(s,z,y,target):
    n=len(y)
    seed=783
    rng=np.random.default_rng(seed)
    idx=np.arange(n)
    folds=[(np.setdiff1d(idx,v),v) for _ in range(2) for v in np.array_split(rng.permutation(n),4)]
    records=[]
    for tag,a,_,lam in _library(s,z,target):
        cv=np.empty((2,n,y.shape[1]))
        for i,(tr,va) in enumerate(folds):
            cv[i//4,va]=_regression(a[tr],a[va],y[tr],lam)
        err=np.abs(cv-y[None,:,:]); fe=np.array([err[i//4,va].mean() for i,(_,va) in enumerate(folds)])
        score=.78*err.mean()+.22*fe.max()
        records.append((score,tag,a,lam,cv))
    records.sort(key=lambda r:r[0]); best=records[0]; blend=best[4].copy(); chosen=[best]
    for i in range(1,8):
        def cost(r):
            err=abs((blend*i+r[4])/(i+1)-y[None,:,:]); return .78*err.mean()+.22*max(err[j//4,va].mean() for j,(_,va) in enumerate(folds))
        row=min(records[:65],key=cost)
        if cost(row) >= cost((None,None,None,None,blend))-.00004:
            break
        blend=(blend*i+row[4])/(i+1); chosen.append(row)
    return np.mean([_regression(a[:n],a[n:],y,lam) for _,_,a,lam,_ in chosen],axis=0)


def run_node(inputs: dict) -> dict:
    s, raw = inputs['__input__'], inputs['node_00']
    z = calibrate(s['panel_a'], np.r_[raw['a'][:, 1:], raw['a_test'][:, 1:]])
    y = np.asarray(s['train']['y']).reshape(-1,1)
    state = np.asarray(s['train']['traj'])
    yp = _predict(s,z,y,0)
    sp = _predict(s,z,state,1)
    return {'test_features':np.c_[yp,sp], 'coefficients':np.eye(1+state.shape[1])}
