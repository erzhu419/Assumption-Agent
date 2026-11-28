"""Panel-calibrated, separately regularized scalar and trajectory prediction."""
from itertools import combinations, combinations_with_replacement
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import erf
from threadpoolctl import threadpool_limits


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _solve(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _maps(st, x):
    b, c = st['panel_b'], st['panel_c']
    u = c['u']
    w = _solve(_aff(b['u']), b['v'])
    z = _aff(x) @ w
    back = (x - w[0]) @ np.linalg.pinv(w[1:])

    def prof(v, forms):
        scores = []
        for fun, bounds in forms:
            def loss(p):
                a = fun(u, p)
                r = (v-a@_solve(a,v)) / np.maximum(.05, 1-np.einsum('ij,ji->i',a,np.linalg.pinv(a)))[:,None]
                return np.mean(r*r)
            o = minimize_scalar(loss, method='bounded', bounds=bounds)
            scores.append((o.fun, o.x, fun))
        _, p, f = min(scores, key=lambda a:a[0])
        a = _solve(f(u,p),v)
        return lambda q:f(q,p)@a

    v = prof(c['v1'], [(lambda q,p:_aff(np.arctan(p*q)),(.3,6)), (lambda q,p:np.c_[_aff(np.arctan(p*q)),q**3],(.3,6)), (lambda q,p:np.c_[_aff(np.arcsinh(p*q)),q**3],(.3,8)), (lambda q,p:np.c_[_aff(np.arctan(p*q)),q*np.roll(q,-1,axis=1)],(.3,6)), (lambda q,p:_aff(q/(1+(p*q)**2)**.25),(.3,8)), (lambda q,p:np.c_[_aff(np.arctan(p*q)),q*q],(.3,6))])
    h = prof(c['v2'], [(lambda q,p:np.c_[_aff(q),np.sqrt(1+p*q*q)-1],(.02,4)), (lambda q,p:np.c_[_aff(q),q*q/(1+p*q*q)],(.001,2)), (lambda q,p:np.c_[_aff(q),np.expm1(-p*q*q)],(.001,2))])
    return z,back,v(x),v(z),h(x),h(z)


def _kernels(st, x, n):
    m,d=x.shape
    raw=np.vstack((st['train']['x'],st['test']['x']))
    z,back,v,vz,w,wz=_maps(st,x)
    labels=[]; mats=[]
    i,j=np.triu_indices(d,1)
    def norm(q):
        q=q-q[:n].mean(0)
        return q/np.maximum(q[:n].std(0),1e-6)
    def gram(q):
        q=norm(q)
        return q@q[:n].T/q.shape[1]
    def add(K,name):
        K=K/max(1e-6,np.trace(K[:n])/n)
        mats.append(K); labels.append(name)
    X=gram(x)
    for coord,q in enumerate((x,z,back,v,vz,w,wz,raw,np.c_[v,w],np.c_[vz,wz])):
        D=q.shape[1]; ij=np.triu_indices(D,1)
        Q=norm(q); L=gram(q)
        ds=(Q[:,None]-Q[None,:n])**2
        for a in [.10,.28,.65,1.4]:
            E=np.exp(-a*ds); av=E.mean(2); pp=(E[:,:,ij[0]]*E[:,:,ij[1]]).mean(2)
            add(np.exp(-a*ds.mean(2))+.2*X,(coord,'rbf'))
            add(av+.3*X,(coord,'add'))
            add(av+.35*pp+.3*X,(coord,'pair'))
        for g in (.08,.3,1.):
            add(L+g*L**2+g*g*L**3+.2*X,(coord,'poly'))
        M=gram(Q[:,ij[0]]*Q[:,ij[1]])
        C=gram(Q[:,ij[0]]*Q[:,ij[1]]/(1+.3*(Q[:,ij[0]]**2+Q[:,ij[1]]**2)))
        U=gram(q**3); S=gram(q*q); Z=gram(q*q/(1+.35*q*q))
        for f in [lambda q:np.sin(q),lambda q:np.tanh(q),lambda q:np.arctan(q),lambda q:np.arcsinh(q),lambda q:erf(q),lambda q:q/np.sqrt(1+q*q)]:
            for k in [.8,1.5,2.15,3.]:
                H=gram(f(k*q)); B=L+H+.16*X
                add(B+.12*U+.13*S+.1*M,(coord,'basis'))
                add(B+.05*U+.08*Z+.15*C,(coord,'sat'))
        if D==4:
            pair=[a+b for a,b in zip(q[:,i].T,q[:,j].T)]+[a-b for a,b in zip(q[:,i].T,q[:,j].T)]
            triples=np.column_stack([q[:,a]*q[:,b]*q[:,c] for a,b,c in combinations(range(4),3)])
            P=gram(np.column_stack([np.sin(k*a) for a in pair for k in (1.,2.)])); T=gram(triples)
            add(L+.25*U+.20*S+.10*M+.04*T+.08*P+.15*X,(coord,'sums'))
            add(L+.12*U+.10*Z+.15*C+.04*T+.25*P+.15*X,(coord,'sums'))
    # Mixed reference spaces retain separately identifiable additive and pairwise effects.
    for a,b in [(0,1),(0,3),(0,5),(0,7),(1,6),(3,5),(4,6),(8,9)]:
        for family in ('basis','sat','pair','rbf','poly'):
            k1=np.mean([K for K,key in zip(mats,labels) if key==(a,family)],axis=0)
            k2=np.mean([K for K,key in zip(mats,labels) if key==(b,family)],axis=0)
            add(.5*(k1+k2),(10,family))
    return mats,labels


def _ensemble(mats, labels, y, t):
    n=len(y)
    # Joint temporal basis retains the two dominant response shapes and also
    # lets observed trajectory noise set the regularization of higher modes.
    V=np.linalg.svd(t,full_matrices=False)[2]
    k=min(3,len(V)); V=V[:k]
    Y=np.column_stack((y,t@V.T)); N=np.eye(n)
    ys=Y-Y.mean(0)
    candidates=[]
    for K,key in zip(mats,labels):
        for lam in (2e-6,3e-5,3e-4,.002,.012,.055):
            cf=cho_factor(K[:n]+3+lam*N,check_finite=False)
            inv=cho_solve(cf,N,check_finite=False)
            A=inv@ys
            e=A/inv.diagonal()[:,None]
            p=(K[n:]+3)@A+Y.mean(0)
            candidates.append((key,e,p))
    E=np.stack([v[1] for v in candidates]); P=np.stack([v[2] for v in candidates]); keys=[v[0] for v in candidates]
    def aggregate(err, pred):
        # Score absolute error and mildly penalize high-error rows, retaining
        # support from distinct coordinate/family combinations.
        a=np.abs(err).reshape(len(E),n,-1).mean(2)
        score=.84*a.mean(1)+.16*np.quantile(a,.8,axis=1)
        order=np.argsort(score); best=score[order[0]]; d={}; ix=[]
        for j in order:
            key=keys[j]
            if d.get(key,0)>=3:continue
            ix.append(j); d[key]=d.get(key,0)+1
            if len(ix)>=22:break
        w=np.exp(-(score[ix]-best)/max(.002,.10*best)); w/=w.sum()
        return np.tensordot(w,pred[ix],axes=1)
    scalar=aggregate(E[:,:,0],P[:,:,0])
    traj=aggregate(E[:,:,1:]@V,P[:,:,1:]@V)
    sep=np.column_stack([aggregate(E[:,:,i],P[:,:,i]) for i in range(1,k+1)])@V
    return scalar,.75*traj+.25*sep


def run_node(inputs: dict) -> dict:
    st=inputs['__input__']; upstream=inputs['node_00']
    x=np.cbrt(np.vstack((upstream['cubic_train'],upstream['cubic_test'])))
    y=np.asarray(st['train']['y'],float).reshape(-1); t=np.asarray(st['train']['traj'],float)
    with threadpool_limits(limits=1,user_api='blas'):
        mats,labels=_kernels(st,x,len(y))
        p,s=_ensemble(mats,labels,y,t)
    return {'y_hat':p,'state_pred':s,'diagnostics_vec':np.array([p.mean(),p.std(),s[:,-1].mean(),s[:,-1].std()])}
