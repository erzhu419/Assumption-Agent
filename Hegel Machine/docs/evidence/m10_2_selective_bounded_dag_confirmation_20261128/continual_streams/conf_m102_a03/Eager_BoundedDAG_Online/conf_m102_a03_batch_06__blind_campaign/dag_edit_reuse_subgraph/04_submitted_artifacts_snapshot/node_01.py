"""Reference-panel calibration and coupled scalar/trajectory regression."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _q(x):
    return np.column_stack([x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)])


def _curve(x, h, k):
    s = h*x
    if k == 0:
        return x*np.exp(-s*s)
    if k == 1:
        return np.sin(s)
    if k == 2:
        return erf(s)
    if k == 3:
        return x/(1+s**4)
    if k == 4:
        return np.tanh(s**3)
    return x/(1+np.abs(s)**3)


def _warp(x, p, kind, flexible):
    h, a = p[:2]
    w = x + a*_curve(x, h, kind)
    if flexible:
        w = w + p[2]*x*x + p[3]*x*np.roll(x,-1,axis=1) + p[4]*x**3
    return _aff(w)


def _solve(a, y, reg):
    if reg == 0:
        return _ls(a,y)
    pen = np.zeros(a.shape[1]); pen[5:] = reg
    return np.linalg.solve(a.T@a + np.diag(pen), a.T@y)


def _panel_modes(u,v):
    candidates=[]
    for kind, scale in enumerate((1.,2.8,1.4,1.,1.,1.)):
        for flexible in (False,True):
            extra = [0.,0.,0.] if flexible else []
            lo = [.25,-3.] + ([-.3,-.3,-.4] if flexible else [])
            hi = [5.,3.] + ([.3,.3,.4] if flexible else [])
            def fun(p):
                d=_warp(u,p,kind,flexible)
                return (d@_ls(d,v)-v).ravel()
            opt=min((least_squares(fun,[scale,b,*extra],bounds=(lo,hi),max_nfev=70) for b in (-1.,-.35,.3)),key=lambda o:o.cost)
            d=_warp(u,opt.x,kind,flexible)
            lev=np.sum(np.linalg.qr(d,mode='reduced')[0]**2,axis=1)
            score=np.linalg.norm(fun(opt.x).reshape(v.shape)/np.maximum(1-lev,.15)[:,None])
            candidates.append((score,kind,flexible,opt.x,np.array(lo),np.array(hi)))
    candidates.sort(key=lambda c:c[0]); return [candidates[0],next(c for c in candidates if not c[2]),next(c for c in candidates if c[1]!=candidates[0][1])]


def _design(x,z,mode):
    a=_aff(z); q=_q(z)
    if mode == 0: return np.c_[a,q]
    if mode == 1: return np.c_[a,q,x]
    if mode == 2: return np.c_[a,q,x,x*x]
    if mode == 3: return np.c_[a,q,x,x*x,z**3]
    if mode == 4: return np.c_[_aff(x),_q(x),x**3]
    if mode == 5: return np.c_[a,q,z**3,*[np.prod(z[:,cols],axis=1) for cols in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]]
    return np.c_[a,q,*[z[:,i]**2*z[:,j] for i in range(4) for j in range(4)]]


def _joint(x,y,xt,u,v,cal,mode):
    _,kind,flexible,p0,lo,hi=cal
    n=len(x); fix=_warp(u,p0,kind,flexible); b0=_ls(fix,v)
    # Profile output coefficients, but constrain the latent transform by the
    # independently measured reference pairs throughout the optimization.
    rows=np.r_[x,u]; initial=np.r_[p0,np.zeros(20)]
    reg=.018 if mode else 0.
    def features(rows,p):
        par=p[:len(p0)]; b=b0+p[len(p0):].reshape(5,4)
        z=_warp(rows,par,kind,flexible)@b
        return _design(rows,z,mode),z
    def loss(p):
        d,z=features(rows,p); d=d[:n]
        coef=_solve(d,y,reg)
        return np.r_[(d@coef-y).ravel(), 2.5*(z[n:]-v).ravel(), .08*(p-initial)]
    solve=least_squares(loss,initial,bounds=(np.r_[lo,np.full(20,-.55)],np.r_[hi,np.full(20,.55)]),max_nfev=48)
    d,_=features(x,solve.x); g,_=features(xt,solve.x)
    return g@_solve(d,y,reg)


def _readouts(x,xt,u,v,y):
    n=len(x); l=np.arange(n); allx=np.r_[x,xt]
    # Predict in a scaled target space so the jointly fitted calibration is
    # not dominated by the eight correlated trajectory columns.
    k=y.shape[1]-1; scale=np.r_[1.,np.full(k,1/np.sqrt(k))]
    targets=y*scale
    cal=_panel_modes(u,v)
    fixed=[]
    for record in cal:
        _,kind,flexible,p,*_=record
        z=_warp(allx,p,kind,flexible)@_ls(_warp(u,p,kind,flexible),v)
        for mode in range(7):
            for reg in ((0.,.08) if mode==0 else (.012,.10,.45)):
                fixed.append((_design(allx,z,mode),reg))
    # Wide but regularized mixing of the raw and nonlinear coordinates.
    z=_warp(allx,cal[0][3],cal[0][1],cal[0][2])@_ls(_warp(u,cal[0][3],cal[0][1],cal[0][2]),v)
    mixed=np.c_[_aff(z),_q(z),allx,allx**3,_q(allx),*[z[:,i:i+1]*allx for i in range(4)]]
    fixed.extend((mixed,r) for r in (.008,.04,.15,.6))
    perm=np.random.default_rng(813).permutation(n)
    folds=[(np.setdiff1d(l,b),b) for b in np.array_split(perm,4)]
    def score(p):
        e=np.abs(p/scale-y)
        # Average across held-out examples, with a small tail penalty.
        return np.array([np.mean(e[:,0])+.08*np.quantile(e[:,0],.9),np.mean(e[:,1:])+.08*np.quantile(e[:,1:],.9)])
    models=[]
    for d,r in fixed:
        p=np.empty_like(y)
        for a,b in folds: p[b]=d[b]@_solve(d[a],targets[a],r)
        models.append((score(p),d[n:]@_solve(d[:n],targets,r)))
    for c in cal[:2]:
        for mode in (0,1,5):
            p=np.empty_like(y)
            for a,b in folds: p[b]=_joint(x[a],targets[a],x[b],u,v,c,mode)
            s=score(p)
            if np.any(s <= np.min([m[0] for m in models],axis=0)*1.12):
                models.append((s,_joint(x,targets,xt,u,v,c,mode)))
    p=np.empty((len(xt),y.shape[1]))
    for label,cols in enumerate((slice(0,1),slice(1,None))):
        best=sorted(models,key=lambda m:m[0][label])[:4]
        s=np.array([m[0][label] for m in best]); w=np.exp(-16*(s/max(s[0],1e-9)-1)); w/=w.sum()
        p[:,cols]=sum(q*m[1][:,cols] for q,m in zip(w,best))
    return p/scale


def run_node(inputs: dict) -> dict:
    study, enc=inputs['__input__'], inputs['node_00']
    panel=study['panel_b']; u,v=(np.asarray(panel[k],float) for k in ('u','v'))
    b=_ls(_aff(u),v)
    def undo(z): return np.linalg.solve(b[1:].T,(np.asarray(z)-b[0]).T).T
    x,xt=undo(enc['z']),undo(enc['zt']); train=study['train']
    pred=_readouts(x,xt,u,v,np.column_stack((train['y'],train['traj'])))
    y,t=pred[:,0],pred[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
