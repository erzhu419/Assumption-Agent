"""Fit the calibrated two-branch response and two-mode trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _align(panel):
    u, v = panel['u'], panel['v']
    # Median of pairwise slopes, followed by inlier least squares.
    i, j = np.triu_indices(len(u), 1)
    dx, dy = u[i] - u[j], v[i] - v[j]
    result = []
    for k in range(u.shape[1]):
        valid = abs(dx[:, k]) > 1e-8
        slope = np.median(dy[valid, k] / dx[valid, k])
        r = v[:, k] - u[:, k] * slope
        r -= np.median(r)
        good = abs(r) <= max(.001, 5 * np.median(abs(r)))
        result.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    return np.asarray(result).T


def _branch(panel, x, k):
    u, v = panel['u'], panel['v'+str(k)]
    if k == 1:
        def features(z, p):
            w = p[0]*z + p[1:]
            return _ones(np.c_[z, w/(1+abs(w))])
        start = [.84, 0., 0., 0., 0.]
    else:
        def features(z, p):
            return _ones(np.c_[z, np.cos(p[0]*z+p[1:])])
        start = [1.27, 0., 0., 0., 0.]
    def residual(p):
        d = features(u, p)
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, start, max_nfev=150).x
    return features(x, p) @ _ls(features(u, p), v)





def _rbf(u, width):
    d = np.maximum(np.sum(u*u,axis=1)[:,None]+np.sum(u*u,axis=1)[None,:]-2*u@u.T,0.)
    return np.exp(-d/(2*width**2))


def _design(study):
    x = np.vstack((study['train']['x'], study['test']['x']))
    a,b = _align(study['panel_a'])
    z = (x-a)/b
    c,d = (_branch(study['panel_c'], z,k) for k in (1,2))
    cx,dx = (_branch(study['panel_c'], x,k) for k in (1,2))
    def cross(c,d):
        return (c[:,:,None]*d[:,None,:]).reshape(len(c),-1)
    scalar = _ones(np.c_[c,d,np.sum(c*d,axis=1)])
    state = _ones(np.c_[c,d,cross(c,d)])
    rawstate = _ones(np.c_[cx,dx,cross(cx,dx)])
    n = len(study['train']['x'])
    def norm(v):
        return (v-v[:n].mean(0))/np.maximum(v[:n].std(0),1.e-6)
    Z,C,D,X,CX,DX = map(norm,(z,c,d,x,cx,dx))
    kernels=[]
    for U in (Z,np.c_[Z,C,D]/np.sqrt(3),np.c_[X,CX,DX]/np.sqrt(3)):
        for h in (.65,1.25,2.5,5.):
            kernels.append(_rbf(U,h))
    affine = _ls(_ones(study['panel_b']['u']),study['panel_b']['v'])
    w = _ones(z) @ affine
    for U in (Z,norm(w), np.c_[Z,norm(w)], np.c_[C,D]):
        for h in (.4,.8,1.6,3.2):
            K = np.array([_rbf(v[:,None],h) for v in U.T]); S=K.sum(0)
            kernels.append(S/len(K))
            kernels.append((S*S-np.sum(K*K,axis=0))/(len(K)*(len(K)-1)))
    tel = float(study['telemetry_vec'][2])
    prim = np.c_[z,c,d,w]
    pairs = np.column_stack([z[:,i]+sign*z[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])
    P=np.c_[prim,pairs]
    F=[]
    for rate in (.84,1.27,tel,3.5):
        q=rate*P
        F.extend((np.sin(q),np.cos(q),q/(1+abs(q)),q/(1+q*q),1/(1+q*q)))
    for u in (z,c,d,w):
        for k in (1.27,tel):
            q=k*u
            for v in (np.sin(q),np.cos(q),q/(1+abs(q)),q/(1+q*q)):
                F.append(cross(z,v))
    F = np.clip(norm(np.column_stack(F)), -15.,15.)
    for U in (F[:,:len(F.T)//2], F):
        kernels.append(U@U.T/U.shape[1])
    return scalar,state,rawstate,kernels,F


def _predict(target, bases, kernels, features, folds, n):
    from scipy.linalg import cho_factor, cho_solve
    q=len(kernels)
    target=np.asarray(target)
    candidates=[]
    def gp(B,I,J,K,r):
        L=cho_factor(K[np.ix_(I,I)] + r*np.eye(len(I)),check_finite=False)
        A=B[I]; Y=target[I]
        Q=cho_solve(L,np.c_[Y,A],check_finite=False)
        m=Y.shape[1]
        beta=_ls(A.T@Q[:,m:]+.005*np.eye(A.shape[1]), A.T@Q[:,:m])
        return B[J]@beta + K[np.ix_(J,I)]@(Q[:,:m]-Q[:,m:]@beta)
    def matching(B,I,J,steps):
        A=B[I]; D=features[I]; G=features[J]; Y=target[I]
        coef=_ls(A,np.c_[Y,D]); H=D-A@coef[:,Y.shape[1]:]
        H=H/np.maximum(np.linalg.norm(H,axis=0),1.e-8)
        Y0=Y-A@coef[:,:Y.shape[1]]; R=Y0.copy(); chosen=[]
        for _ in range(steps):
            p=np.linalg.norm(H.T@R,axis=1)
            p[chosen]=-1
            chosen.append(int(np.argmax(p)))
            E=np.c_[A,D[:,chosen]]
            coef1=_ls(E,Y); R=Y-E@coef1
        return np.c_[B[J],G[:,chosen]]@coef1
    all_idx=np.arange(n)
    test=np.arange(n,len(bases[0]))
    for B in bases:
        for k in range(q):
            for r in (.001,.03,.3):
                candidates.append((B,k,r))
        for steps in (3,7,12):
            candidates.append((B,-1,steps))
    def evaluate(c,I,J):
        B,k,r=c
        return gp(B,I,J,kernels[k],r) if k>=0 else matching(B,I,J,r)
    results=[]
    for c in candidates:
        error=[]
        for J in folds:
            I=np.setdiff1d(all_idx,J,assume_unique=True)
            error.append(np.mean(abs(evaluate(c,I,J)-target[J])))
        v=np.mean(error)+.20*np.max(error)
        results.append(v if np.isfinite(v) else np.inf)
    rank=np.argsort(results)
    keep=[]; seen=set(); best=max(results[rank[0]],1.e-10)
    for k in rank:
        B,index,r=candidates[k]
        key=(id(B),index)
        if key in seen: continue
        if len(keep)>=8 or results[k]>1.25*best: break
        seen.add(key);keep.append(k)
    weights=np.exp(-12*(np.array(results)[keep]/best-1))
    return np.average([evaluate(candidates[k],all_idx,test) for k in keep],axis=0,weights=weights)


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; train=s['train']; n=len(train['x'])
    scalar,state,rawstate,kernels,features=_design(s)
    rng=np.random.default_rng(1709)
    folds=[np.sort(j) for j in np.array_split(rng.permutation(n),4)]
    y=_predict(np.asarray(train['y'])[:,None], [scalar,scalar[:,:1]],kernels,features,folds,n)[:,0]
    t=_predict(np.asarray(train['traj']),[state,rawstate,scalar,scalar[:,:1]],kernels,features,folds,n)
    return {'y_hat':y,'unprojected':t}
