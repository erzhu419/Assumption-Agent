"""Calibrate the inputs and predict both outputs with a shared residual kernel."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_basis(x):
    return np.stack((np.ones_like(x), x, x**3,
                     (x-1)*np.roll(x, -1, axis=1)), axis=2)


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(v, axis=1) < 5
    a = _calibration_basis(u[keep])
    w = np.stack([_fit(a[:, i], v[keep, i]) for i in range(4)])
    return np.stack([least_squares(lambda x: (_calibration_basis(x[None])*w).sum(2)[0]-r,
                                   r, gtol=1e-11).x for r in raw])


def _bilinear(x):
    a, b, c, d = x.T
    return _bias(np.column_stack((a*b, a*d, b*c, c*d)))


def _branches(panel, x):
    u = panel['u']
    c = _bilinear(x) @ _fit(_bilinear(u), panel['v1'])
    d = _bias(np.cos(2.4*x)) @ _fit(_bias(np.cos(2.4*u)), panel['v2'])
    return c, d


def _features(study, raw, x, c, d):
    from itertools import combinations_with_replacement
    bpanel = study['panel_b']
    wb = _fit(_bias(bpanel['u']), bpanel['v'])
    b = _bias(x) @ wb
    u = study['panel_c']['u']
    wc = _fit(_bilinear(u), study['panel_c']['v1'])
    wd = _fit(_bias(np.cos(2.4*u)), study['panel_c']['v2'])
    def cross(a, b):
        return (a[:, :, None]*b[:, None, :]).reshape(len(x), -1)
    groups = []
    for f in (raw, x, b):
        for j in (2,3,4):
            groups.append(np.column_stack([np.prod(f[:, k],axis=1) for k in combinations_with_replacement(range(4),j)]))
        for w in (1.2, 2.4, 4.8):
            groups.append(np.column_stack((np.sin(w*f),np.cos(w*f))))
        for j in (-1,1):
            m=f+np.roll(f,j,axis=1)
            groups.append(np.column_stack((np.cos(2.4*m),np.sin(2.4*m))))
    groups.extend([cross(c,d),cross(x,c),cross(x,d),cross(raw,c),cross(raw,d),cross(b,c),cross(b,d),cross(c,c),cross(d,d)])
    for f in (c, d, c+d, c-d, c*d, c+b, d+b, b, raw, x+c, x+d):
        groups.append(np.column_stack((np.sin(2.4*f),np.cos(2.4*f),np.tanh(f),f/(1+f*f))))
    for f in (b,raw,d,c,(x+b)/2,x-b):
        groups.append(_bilinear(f) @ wc)
        groups.append(_bias(np.cos(2.4*f)) @ wd)
    return groups


def _joint(study, raw, x, c, d, target):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    n = len(target)
    h = _bias(np.column_stack((c,d)))
    groups = _features(study, raw,x,c,d)
    K = []
    for a in (raw,x,np.c_[c,d]):
        q=cdist(a/a[:n].std(0),a[:n]/a[:n].std(0),'sqeuclidean')/a.shape[1]
        for w in (.06,.2,.5,1.25,3.):
            K.append(np.exp(-w*q))
    for a in groups:
        a=(a-a[:n].mean(0))/np.maximum(a[:n].std(0),1e-9)
        K.append(a@a[:n].T/a.shape[1])
    K.append(np.vstack((np.eye(n),np.zeros((len(raw)-n,n)))))
    K=np.asarray(K)
    H=h[:n]; q=np.linalg.qr(H,mode='complete')[0][:,H.shape[1]:]
    Pr=np.array([q@(q.T@k[:n]@q)@q.T for k in K]); flat=Pr.reshape(len(K),-1)
    Y=target.copy(); Y[:,1:]/=np.sqrt(Y.shape[1]-1)
    prior=np.full(len(K),-6.); prior[:15]=-4; prior[2]=0; prior[-1]=-9
    def fit_cov(p, predict=False):
        w=np.exp(p)
        if predict:
            k=np.einsum('i,ijk->jk',w,K)
            C=cho_factor(k[:n]); a=cho_solve(C,H); beta=_fit(H.T@a,a.T@target)
            return h[n:]@beta+k[n:]@cho_solve(C,target-H@beta)
        # Augment the projected covariance with identity on the removed trend space.
        C=cho_factor((w@flat).reshape(n,n)+J)
        P=cho_solve(C, np.eye(n))-J
        AY=P@Y; v=P.diagonal(); e=AY/v[:,None]; val=np.mean(np.sum(e*e,axis=1))
        den=2*e/n/v[:,None]
        G=-(P@den)@AY.T+(P*(np.sum(den*e,axis=1))[None,:])@P
        reg=.00006*np.mean((p-prior)**2)
        return val+reg, w*(flat@G.ravel())+.00012*(p-prior)/len(p)
    J=np.eye(n)-q@q.T
    p=minimize(fit_cov,prior,jac=True,method='L-BFGS-B',bounds=[(-17,6)]*(len(K)-1)+[(-17,0)],options={'maxiter':125,'ftol':1e-10}).x
    return fit_cov(p,True)


def _baseline(raw, h, target):
    n=len(target)
    trend=h@_fit(h[:n],target)
    f=raw/raw[:n].std(0)
    k=np.exp(-.125*cdist(f,f[:n],'sqeuclidean'))
    return trend[n:]+k[n:]@np.linalg.solve(k[:n]+.0001*np.eye(n),target-trend[:n])


def solve_batch(study):
    train=study['train']; raw=np.concatenate((train['x'],study['test']['x']))
    x=_calibrate(study['panel_a'],raw); c,d=_branches(study['panel_c'],x)
    target=np.column_stack((train['y'],train['traj']))
    h=_bias(np.column_stack((c,d)))
    pred=_joint(study,raw,x,c,d,target)
    y,state=pred[:,0],pred[:,1:]
    end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
