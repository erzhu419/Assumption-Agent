import numpy as np
from scipy.optimize import least_squares, root
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _coords(study):
    u, v = [np.asarray(study['panel_a'][k]) for k in ('u','v')]
    r = np.linalg.norm(v-u, axis=1)
    med = np.median(r)
    keep = r <= med + max(1., 8*np.median(abs(r-med)))
    u, v = u[keep], v[keep]
    def features(q, p, j):
        return np.c_[np.ones(len(q)), q[:,j], np.tanh(p[0]*q[:,j]), np.sin(p[1]*q[:,(j+2)%4])]
    def residual(p):
        return np.column_stack([f@_ls(f,v[:,j])-v[:,j] for j in range(4) for f in [features(u,p,j)]]).ravel()
    p = least_squares(residual, [1.8,2.4], bounds=([.7,1.4],[3.,3.4])).x
    w = np.array([_ls(features(u,p,j),v[:,j]) for j in range(4)])
    bias, gain, amp, cross = w.T
    n = len(study['train']['x'])
    x = np.vstack([study['train']['x'],study['test']['x']])
    def forward(q):
        return bias+gain*q+amp*np.tanh(p[0]*q)+cross*np.sin(p[1]*np.roll(q,2))
    q = np.array([root(lambda q: forward(q)-r,(r-bias)/(gain+amp)).x for r in x])
    return q,n


def _features(q, b, c, t):
    o = np.ones(len(q)); lin = np.c_[o,q,b,c]
    quad = np.column_stack([np.prod(q[:,i],axis=1) for m in range(3) for i in combinations_with_replacement(range(4),m)])
    gs = [1/(1+t[i]*q[:,i]**2) for i in range(4)]
    for i in range(4):
        for k in t[:2]:
            gs.append(1/(1+k*q[:,i]**2))
    for i in range(4):
        for j in range(i+1,4):
            for r in [-1.,.5,1.]:
                for k in t[:2]: gs.append(1/(1+k*(q[:,i]+r*q[:,j])**2))
    for g in list(gs)[:12]:
        yield np.c_[o,c,b*g[:,None]]
        for m in [2,3]:
            yield np.c_[o,c,b*g[:,None],b*q[:,0,None]**m*g[:,None]]
        yield np.c_[quad,lin*g[:,None]]
    for g in gs[12:]:
        yield np.c_[o,c,b*g[:,None]]
        yield np.c_[quad,np.c_[o,q]*g[:,None]]
    for i in range(4):
        for j in range(4):
            for g in (gs[i]*gs[j],gs[i]):
                yield np.c_[o,c,b*g[:,None],b*q[:,j,None]**3*g[:,None],b*gs[j][:,None]]
    yield quad
    yield np.c_[quad,b,c]


def _scalar(q,b,c,t,y):
    n=len(y)
    candidates=[]
    for f in _features(q,b,c,t):
        train=f[:n]; pinv=np.linalg.pinv(train); w=pinv@y
        lev=np.einsum('ij,ji->i',train,pinv)
        err=np.mean(abs((y-train@w)/np.maximum(1-lev,.025)))
        candidates.append((err,f[n:]@w))
    lin=np.c_[np.ones(len(q)),q,b,c]; L=lin/np.maximum(lin[:n].std(0),.07); L[:,0]=1
    for Q in [q, np.c_[q,b/4], np.c_[q,b/2]]:
        d=cdist(Q,Q,'sqeuclidean')
        for length in [.5,1.,2.,4.]:
            R=np.sqrt(d)/length
            for K in [np.exp(-.5*R**2),(1+R+R**2/3)*np.exp(-R)]:
                K=K+L@L.T
                for noise in [1e-5,.0005,.01,.05]:
                    A=np.linalg.inv(K[:n,:n]+noise*np.eye(n)); w=A@y
                    candidates.append((np.mean(abs(w/A.diagonal())),K[n:,:n]@w))
    candidates.sort(key=lambda v:v[0]); best=candidates[0][0]
    top=[(e,p) for e,p in candidates[:8] if e<=1.15*best]
    return np.average([p for _,p in top],axis=0,weights=[(best/max(e,1e-12))**8 for e,_ in top])


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    q,n = _coords(study)
    a,k,e = upstream['p']
    h = q*np.exp(e*np.logaddexp(0,k*(np.log(a)+np.log(np.maximum(abs(q),1e-100)))))
    b = np.c_[np.ones(len(q)),q,h]@upstream['w']
    c = b*np.roll(q,-1,axis=1)
    tr = study['train']
    state = np.c_[np.ones(len(q)),b,c.sum(1)]
    state = state[n:]@_ls(state[:n],tr['traj'])
    y = _scalar(q,b,c,study['telemetry_vec'],tr['y'])
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
