from itertools import combinations, combinations_with_replacement, product

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    j = np.roll(u, -1, axis=1)
    designs = [np.column_stack((np.ones(len(u)), p, q, p*q, p**3))
               for p, q in zip(u.T, j.T)]
    coeff = np.empty((5, u.shape[1]))
    weight = np.ones(len(u))
    for _ in range(24):
        residual = np.empty_like(v)
        for i, a in enumerate(designs):
            coeff[:, i] = np.linalg.lstsq(a * weight[:, None], v[:, i] * weight,
                                         rcond=None)[0]
            residual[:, i] = v[:, i] - a @ coeff[:, i]
        radius = np.linalg.norm(residual, axis=1)
        scale = max(np.median(radius), 1e-8)
        weight = 1 / np.hypot(1, radius / scale)

    b, d, e, f, c = coeff
    idx = np.arange(u.shape[1])
    initial = np.diag(d)
    initial[idx, (idx + 1) % len(idx)] = e
    z = np.linalg.solve(initial, (x - b).T).T
    for _ in range(16):
        q = np.roll(z, -1, axis=1)
        r = b + d*z + e*q + f*z*q + c*z**3 - x
        jac = np.zeros((len(x), len(idx), len(idx)))
        jac[:, idx, idx] = d + f*q + 3*c*z*z
        jac[:, idx, (idx + 1) % len(idx)] = e + f*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step / np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return z


def _predict(u, y, z, feature):
    return feature(z) @ np.linalg.lstsq(feature(u), y, rcond=None)[0]


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _saturating(u):
    return np.column_stack((_affine(u), np.arctan(2.4 * u)))


def _square(u):
    return np.column_stack((_affine(u), u*u))


def _polynomial(x, degree):
    return np.column_stack([np.prod(x[:, j], axis=1) for k in range(1, degree+1)
                            for j in combinations_with_replacement(range(x.shape[1]), k)])


def _dictionary(z, f, c1, c2):
    out = [_polynomial(w, 4) for w in (z, f)]
    for w in (z, f, c1, c2):
        a, b = np.triu_indices(w.shape[1], 1)
        pairs = np.column_stack((w, w[:, a]+w[:, b], w[:, a]-w[:, b]))
        out.extend((np.sin(1.5*pairs), np.cos(1.5*pairs), np.tanh(1.5*pairs),
                    pairs/(1+pairs*pairs), np.log1p(pairs*pairs)))
        out.append(np.column_stack([w[:, i]*w[:, j]*w[:, k]
                                    for i,j,k in combinations(range(w.shape[1]),3)]))
    for w in (f, c1, c2):
        out.append((z[:,:,None]*w[:,None,:]).reshape(len(z),-1))
        out.append((z[:,:,None]**2*w[:,None,:]).reshape(len(z),-1))
    return np.column_stack(out)


def _fit_predictions(z, f, c1, c2, a, b, target):
    def scale(w):
        return (w-w[a].mean(0))/np.maximum(w[a].std(0), 1e-8)
    z, f, c1, c2 = map(scale,(z,f,c1,c2))
    mean, sd = target.mean(0), np.maximum(target.std(0),1e-8)
    Y = (target-mean)/sd
    I = np.eye(len(a))
    predictions = []
    def kernel(K, lam):
        A = K[np.ix_(a,a)]
        B = K[np.ix_(b,a)]
        predictions.append(B @ cho_solve(cho_factor(A+lam*I,check_finite=False),Y,check_finite=False))
    qz, qf = [_polynomial(w,3) for w in (z,f)]
    base = scale(np.c_[qz,qf]); Q=base@base.T/base.shape[1]
    for w in (z, f, np.c_[z,f]/np.sqrt(2), np.c_[z,f,.35*c1,.35*c2]/np.sqrt(2)):
        norm = np.sum(w*w,axis=1)
        d2 = np.maximum(norm[:,None]+norm[None,:]-2*w@w.T,0)
        for length in (1.15,1.9,3.0):
            r=np.sqrt(d2)/length
            kernel(Q+np.exp(-.5*r*r),.0015)
            if length==1.9:
                kernel(Q+(1+r+r*r/3)*np.exp(-r),.002)
    for w in (z, f):
        for degree in (3,4):
            P = scale(_polynomial(w,degree)); K=P@P.T/P.shape[1]
            kernel(K,.001)
            kernel(K,.015)
    for raw in (np.c_[qz,qf], _dictionary(z,f,c1,c2)):
        F=scale(raw); A=F[a]; B=F[b]; P=F.shape[1]
        for noise in (.009,.04):
            pred=[]
            for y in Y.T:
                weights=np.full(P,1/P)
                for _ in range(36):
                    aw=A*weights
                    chol=cho_factor(aw@A.T+noise*I,check_finite=False)
                    work=cho_solve(chol,np.column_stack((y,A)),check_finite=False)
                    u=A.T@work[:,0]
                    h=np.sum(A*work[:,1:],axis=0)
                    weights *= np.clip(np.abs(u)/np.sqrt(np.maximum(h,1e-10)),.45,2.2)**.75
                    weights=np.maximum(weights,1e-10)
                coef=weights*(A.T@cho_solve(cho_factor((A*weights)@A.T+noise*I,check_finite=False),y,check_finite=False))
                pred.append(B@coef)
            predictions.append(np.column_stack(pred))
    return np.stack(predictions,axis=-1)*sd[None,:,None]+mean[None,:,None]


def _response_features(z, f, c1, c2, y, traj):
    n=len(y)
    basis=np.linalg.svd(traj,full_matrices=False)[2][:2]
    target=np.column_stack((y,traj@basis.T))
    mix=[]
    for b in np.array_split(np.random.default_rng(481).permutation(n),5):
        a=np.setdiff1d(np.arange(n),b)
        # Use the fold's trajectory coordinates and rotate the two predictions
        # into the common coordinates only after fitting.
        sub=np.linalg.svd(traj[a],full_matrices=False)[2][:2]
        p=_fit_predictions(z,f,c1,c2,a,b,np.c_[y[a],traj[a]@sub.T]); p[:,1:]=np.einsum('bjm,jk->bkm',p[:,1:],sub@basis.T)
        mix.append((b,p))
    oof=np.empty((n,*mix[0][1].shape[1:]))
    for b,p in mix: oof[b]=p
    a=np.arange(n)
    pred=_fit_predictions(z,f,c1,c2,a,np.arange(len(z)),target)
    result=[]
    for j in range(target.shape[1]):
        A=oof[:,j]-target[:,j,None]
        error=np.sqrt(np.mean(A*A,axis=0)); best=np.argsort(error)[:7]
        A=A[:,best]; unit=np.ones((1,len(best)))
        weights=nnls(np.r_[A, 2*np.linalg.norm(A)*unit, .12*np.linalg.norm(A)*np.eye(len(best))],
                     np.r_[np.zeros(n),2*np.linalg.norm(A),np.zeros(len(best))])[0]
        weights/=max(weights.sum(),1e-12)
        result.append(pred[:,j,best]@weights)
    return np.column_stack(result)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']; train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    fd = _predict(b['u'], b['v'], z, _affine)
    c1 = _predict(c['u'], c['v1'], z, lambda u: np.c_[_affine(u), np.arcsinh(3.46*u)])
    c2 = _predict(c['u'], c['v2'], z, lambda u: np.c_[_affine(u), np.log1p(.36*u*u)])
    p = _response_features(z, fd, c1, c2, train['y'], train['traj'])
    g = np.c_[np.ones(len(z)), p]
    return {'g': g, 'fd': .03*fd, 'cd': .03*np.c_[c1,c2], 'y': train['y'], 'traj': train['traj']}
