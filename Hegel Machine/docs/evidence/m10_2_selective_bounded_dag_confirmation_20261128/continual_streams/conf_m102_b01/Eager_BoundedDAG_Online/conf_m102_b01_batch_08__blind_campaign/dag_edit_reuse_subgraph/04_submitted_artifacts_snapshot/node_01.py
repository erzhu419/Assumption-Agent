import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from itertools import combinations_with_replacement


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _cross(x):
    return np.column_stack([x[:, i]*x[:, j] for i,j in [(0,1),(0,3),(1,2),(2,3)]])


def _cal(s, w):
    b, c = s['panel_b'], s['panel_c']
    a = _fit(_aff(b['u']), b['v'])
    u = _fit(a[1:].T, (w-a[0]).T).T
    def design(x, g):
        return np.column_stack((_aff(x), x**3/(1+g*x*x)))
    def residual(g):
        d = design(b['u'], g)
        return (d @ _fit(d,b['v']) - b['v']).ravel()
    gain = least_squares(residual, np.full(u.shape[1], 1.25), bounds=(.01, 5.)).x
    d = design(b['u'], gain)
    W = design(u, gain) @ _fit(d,b['v'])
    def c1(x):
        return np.column_stack((np.ones(len(x)),_cross(x)))
    def c2(x):
        return np.column_stack((_aff(x), *[x[:, i]*x[:, j] for i,j in combinations_with_replacement(range(x.shape[1]), 2)]))
    C1 = c1(u) @ _fit(c1(c['u']),c['v1'])
    C2 = c2(u) @ _fit(c2(c['u']),c['v2'])
    return u, W, C1, C2


def _poly(x, degs):
    return np.column_stack([np.prod(x[:, ix], axis=1) for d in degs for ix in combinations_with_replacement(range(x.shape[1]), d)])


def _design(s, w):
    u, b, c, d = _cal(s,w)
    n = len(s['train']['y'])
    b0 = b - _fit(_aff(s['panel_b']['u']),s['panel_b']['v'])[0]
    views = [b, np.arctan(1.1*b), np.arctan(2*b), np.arcsinh(2*b), b0, u, c, d, np.c_[b,c,d], np.c_[np.arctan(1.1*b),c,d], np.c_[b,u]]
    specs = []
    for j,z in enumerate(views):
        z = (z-z[:n].mean(0))/np.maximum(z[:n].std(0),1e-7)
        p = z @ z.T/z.shape[1]
        dist = cdist(z,z,'sqeuclidean')/z.shape[1]
        for deg in [2,3,4]:
            specs.append(((1+p)**deg, [.001,.02,.12, .6]))
        for width in [.8,2.5,7,16]:
            specs.append((np.exp(-dist/width), [1e-5,.0003,.006,.04]))
    uni=[]
    cross=[]
    for z in [b,np.arctan(1.1*b),np.arctan(2.4*b),np.arcsinh(2*b),u,c,d]:
        uni.append(_poly(z,[1,2,3]))
        cross.append(z)
    base = np.column_stack(uni+[_poly(b,[4]),_poly(np.arctan(1.1*b),[4]),_poly(np.c_[cross[1],c,d],[2])])
    extras=[]
    for z in [b,c,d]:
        for a in [b,u,np.arctan(1.1*b)]:
            extras.append((a[:,:,None]*z[:,None,:]).reshape(len(a),-1))
    z=np.column_stack([base,*extras]); z=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),1e-7)
    return specs,z


def _predict_kernel(k, r, target, tr, te):
    a = k[np.ix_(tr,tr)].copy(); a.flat[::len(tr)+1] += r
    mu=target[tr].mean(0)
    return mu + k[np.ix_(te,tr)] @ cho_solve(cho_factor(a,check_finite=False),target[tr]-mu,check_finite=False)


def _sparse(z, target, tr, te, strength):
    x = z[tr]; x = x - x.mean(0); v=z[te]-z[tr].mean(0)
    y=target[tr]; mu=y.mean(0); y=y-mu
    scale=np.maximum(np.std(y,axis=0),1e-7); y=y/scale
    g=np.full(x.shape[1],1/x.shape[1]); eye=np.eye(len(tr))*strength
    for _ in range(20):
        q=x*g; A=cho_factor(q@x.T+eye,check_finite=False)
        beta=q.T@cho_solve(A,y,check_finite=False)
        h=np.maximum(1-g*np.sum(x*cho_solve(A,x,check_finite=False),axis=0),.015)
        g=.35*g+.65*np.clip(np.mean(beta**2,axis=1)/h,1e-7,4.)
    return mu+scale*(v@beta)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; target=np.column_stack((s['train']['y'],s['train']['traj']))
    n=len(target); tr=np.arange(n)
    specs,z=_design(s,inputs['node_00']['w']); te=np.arange(n,len(z))
    folds=[(np.setdiff1d(tr,j),j) for j in np.array_split(np.random.default_rng(891).permutation(n),4)]
    def score(p):
        e=np.mean(np.abs(p-target),axis=0)
        return np.array([e[0],e[1:].mean()])
    models=[]
    for k,reg in specs:
        for r in reg:
            p=np.empty_like(target)
            for i,j in folds: p[j]=_predict_kernel(k,r,target,i,j)
            models.append((score(p),_predict_kernel(k,r,target,tr,te)))
    for r in [.002,.01,.035,.10]:
        p=np.empty_like(target)
        for i,j in folds: p[j]=_sparse(z,target,i,j,r)
        models.append((score(p),_sparse(z,target,tr,te,r)))
    scores=np.array([v[0] for v in models]); preds=np.stack([v[1] for v in models])
    result=[]
    for col in range(2):
        idx=np.argsort(scores[:,col])[:18]
        weight=np.exp(-30*(scores[idx,col]/scores[idx[0],col]-1));weight/=weight.sum()
        result.append(np.tensordot(weight,preds[idx],axes=1))
    y=result[0][:,0]; state=result[1][:,1:]; tail=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}

# Fit separate covariance components for scalar response and trajectory amplitude.
# Response fitting, nonlinear feature selection, and ensemble weighting use train rows only.
_fixed_run = run_node

def _covariances(s, w):
    u,b,c,d=_cal(s,w)
    n=len(s['train']['y'])
    def norm(x):
        x=x-x[:n].mean(0)
        return x/np.maximum(x[:n].std(0),1e-8)
    groups=[]
    def add(f):
        f=norm(f); groups.append((f@f.T)/f.shape[1])
    add(u); add(c); add(d); add(_poly(u,[2])); add(_poly(c,[2])); add(_poly(d,[2]))
    add(_poly(np.c_[u,b],[2])); add(_poly(np.c_[b,c],[2])); add(_poly(np.c_[b,d],[2]));
    for g in [.65,1.35,2.6]:
        z=np.arctan(g*b)
        add(z)
        for j in range(z.shape[1]):
            q=z[:,j:j+1]; add(np.c_[q,q*q,q**3,q**4,q**5])
        add(_poly(z,[2])); add(_poly(z,[3])); add(_poly(z,[4]));
        for h in [u,c,d]:
            add((z[:,:,None]*h[:,None,:]).reshape(len(z),-1))
        zz=norm(z); dist=cdist(zz,zz,'sqeuclidean')/zz.shape[1]
        for width in [1.,5.,15.]: groups.append(np.exp(-dist/width))
    for z in [b,np.tanh(1.5*b),np.arcsinh(2*b)]:
        add(_poly(z,[1,2,3])); add(_poly(z,[4,5]))
    return np.stack(groups)


def _evidence(ks,y,tr,te,penalty):
    from scipy.optimize import minimize
    from scipy.linalg import cho_factor, cho_solve
    kt=ks[:,tr[:,None],tr].copy()
    kt=np.concatenate((kt,np.eye(len(tr))[None],np.ones((1,len(tr),len(tr)))),axis=0)
    cross=ks[:,te[:,None],tr]
    y=y[tr]; mean=y.mean(); scale=max(y.std(),1e-8); yc=(y-mean)/scale
    mat=kt.reshape(len(kt),-1)
    prior=np.r_[np.full(len(ks),-4.5),-5.,-2.]
    eye=np.eye(len(tr))
    def obj(p):
        v=np.exp(p)
        cf=cho_factor((v@mat).reshape(len(tr),len(tr)),check_finite=False)
        a=cho_solve(cf,yc,check_finite=False)
        inv=cho_solve(cf,eye,check_finite=False)
        grad=.5*v*(mat@(inv-np.outer(a,a)).ravel())+penalty*(p-prior)
        f=.5*yc@a+np.log(np.diag(cf[0])).sum()+.5*penalty*np.sum((p-prior)**2)
        return f,grad
    op=minimize(obj,prior,jac=True,method='L-BFGS-B',bounds=[(-13,4)]*len(ks)+[(-11,0),(-8,3)],options={'maxiter':110,'ftol':1e-8})
    v=np.exp(op.x)
    a=cho_solve(cho_factor((v@mat).reshape(len(tr),len(tr)),check_finite=False),yc,check_finite=False)
    return mean+scale*((v[:-2]@cross.reshape(len(ks),-1)).reshape(len(te),len(tr))@a+v[-1]*a.sum())


def _structural(s,w):
    k=_covariances(s,w)
    y=s['train']['y']; t=s['train']['traj']; n=len(y)
    ix=np.arange(n); te=np.arange(n,k.shape[1]); folds=[(np.setdiff1d(ix,j),j) for j in np.array_split(np.random.default_rng(381).permutation(n),4)]
    def fit(tr,te,p):
        _,_,v=np.linalg.svd(t[tr],full_matrices=False)
        scalar=_evidence(k,y,tr,te,p)
        amp=_evidence(k,t@v[0],tr,te,p)
        return scalar,amp[:,None]*v[0]
    err=[]
    penalties=[.057,.19,.60]
    for p in penalties:
        es=[]
        for tr,va in folds:
            py,pt=fit(tr,va,p)
            es.append([np.mean(abs(py-y[va])),np.mean(abs(pt-t[va]))])
        err.append(np.mean(es,axis=0))
    err=np.array(err)
    weight=np.exp(-15*(err/np.min(err,axis=0)-1));weight/=weight.sum(0)
    ps=[fit(ix,te,p) for p in penalties]
    return sum(wi*pr[0] for wi,pr in zip(weight[:,0],ps)),sum(wi*pr[1] for wi,pr in zip(weight[:,1],ps))


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; w=inputs['node_00']['w']
    base=_fixed_run(inputs)
    y,t=_structural(s,w)
    y=.72*y+.28*base['y_hat']; t=.72*t+.28*base['state_pred']
    tail=t[:,-1]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}
