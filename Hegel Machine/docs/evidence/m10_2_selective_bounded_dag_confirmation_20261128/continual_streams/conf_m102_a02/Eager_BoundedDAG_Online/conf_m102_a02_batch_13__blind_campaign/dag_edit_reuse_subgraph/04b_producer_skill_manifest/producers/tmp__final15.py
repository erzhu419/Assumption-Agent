from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def coordinates(study, raw):
    x = np.vstack((raw['a'][:, 1:], raw['a_test'][:, 1:]))
    p = study['panel_a']
    u, v = np.asarray(p['u']), np.asarray(p['v'])
    successor = np.roll(u, -1, axis=1)
    coefficients = []
    for a, b, target in zip(u.T, successor.T, v.T):
        d = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        c = least_squares(lambda c: d@c-target, fit(d, target), loss='cauchy', f_scale=.002).x
        m = np.abs(d@c-target) < .002
        coefficients.append(fit(d[m], target[m]) if m.sum() >= d.shape[1] else c)
    c = np.asarray(coefficients).T
    def forward(z):
        w = np.roll(z, -1)
        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3
    def jac(z):
        j = np.diag(c[1] + c[3]*np.roll(z, -1) + 3*c[4]*z*z)
        i = np.arange(len(z))
        j[i, (i+1)%len(z)] = c[2]+c[3]*z
        return j
    z = np.array([least_squares(lambda z: forward(z)-row, row, jac=jac, gtol=1e-11).x for row in x])
    p = study['panel_b']
    def design(a, w):
        return np.column_stack((np.ones(len(a)), np.cos(w*a)))
    def loss(w):
        a = design(p['u'], w)
        return np.mean((a@fit(a, p['v'])-p['v'])**2)
    w = minimize_scalar(loss, bounds=(2.,3.), method='bounded').x
    b = design(z, w)
    return b[:,1:], b@fit(design(p['u'],w),p['v'])


def endpoint(u, g, y, spec):
    base, numerator, penalty, freq, shift, kernel = spec
    n = len(y)
    a = np.column_stack((np.ones(len(g)), g))
    cross = g[:,:2]*g[:,2:]
    quad = np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(g.shape[1]),2)])
    d = (u[:,0]-shift*u[:,2])**2
    def make(w):
        r = 1/(1+w*d)
        terms = {'one': u[:,2:3], 'pair': u[:,[0,2]], 'all': u, 'const': np.column_stack((np.ones(len(u)),u[:,2])), 'cross': np.column_stack((u[:,2],cross))}[numerator] if numerator != 'none' else np.empty((len(u),0))
        core = np.column_stack((a, cross, r[:,None]*terms))
        if base == 'cross': return core, core.shape[1]
        return np.column_stack((core,quad)), core.shape[1]
    def train(w, loo=False):
        X, k = make(w)
        if base == 'cross' or penalty == 0:
            if not loo: return X, fit(X[:n],y)
            U,S,_ = np.linalg.svd(X[:n],full_matrices=False)
            U = U[:,S>S[0]*1e-9]; h=(U**2).sum(1); e=y-U@(U.T@y)
        else:
            M = X[:n].T@X[:n] + np.diag(np.r_[np.zeros(k),np.full(X.shape[1]-k,penalty)])
            L = np.linalg.pinv(M,rcond=1e-12)@X[:n].T
            c=L@y
            if not loo: return X,c
            h=(X[:n]*L.T).sum(1); e=y-X[:n]@c
        return np.mean((e/np.maximum(1-h,.02))**2)
    if freq == 0:
        ws = [.10,.22,.4,.65,1.,2.,4.]
        w0=min(ws,key=lambda w: train(w,True)); j=ws.index(w0)
        w = minimize_scalar(lambda w:train(w,True),bounds=(ws[max(0,j-1)],ws[min(len(ws)-1,j+1)]),method='bounded',options={'xatol':.001}).x
    else:
        w=freq
    X,c=train(w)
    if kernel:
        # A small smooth residual correction, projected off the structural trend.
        from scipy.spatial.distance import cdist
        X, k = make(w)
        X = X[:,:k]
        U, S, _=np.linalg.svd(X[:n],full_matrices=False)
        U=U[:,S>S[0]*1e-9]; J=np.eye(n)-U@U.T
        coords=u/np.maximum(u[:n].std(0),1e-8)
        dist=cdist(coords[:n],coords,'sqeuclidean')
        best=(np.inf,None,None)
        for width in [.015,.05,.15,.4]:
            K=np.exp(-width*dist); A=K[:,:n]; lam,V=np.linalg.eigh(J@A@J); V=J@V; Vy=V.T@y
            for reg in [.0001,.003,.03,.15]:
                R=V/(np.maximum(lam,0)+reg); alpha=R@Vy
                err=np.mean(np.abs(alpha/np.maximum((R*V).sum(1),1e-9)))
                if err < best[0]: best=(err,K,alpha)
        _,K,alpha=best
        return np.column_stack((X,K.T)), np.r_[fit(X[:n],y-K[:,:n]@alpha),alpha]
    return X,c


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    u,g = coordinates(study,raw)
    y, traj = study['train']['y'], study['train']['traj']
    n=len(y)
    X,c=endpoint(u,g,y,SPEC)
    a=np.column_stack((np.ones(len(g)),g))
    features=np.column_stack((a,X))
    coefficients=np.zeros((features.shape[1],1+traj.shape[1]))
    coefficients[a.shape[1]:,0]=c
    coefficients[:a.shape[1],1:]=fit(a[:n],traj)
    return {'test_features':features[n:], 'coefficients':coefficients}
