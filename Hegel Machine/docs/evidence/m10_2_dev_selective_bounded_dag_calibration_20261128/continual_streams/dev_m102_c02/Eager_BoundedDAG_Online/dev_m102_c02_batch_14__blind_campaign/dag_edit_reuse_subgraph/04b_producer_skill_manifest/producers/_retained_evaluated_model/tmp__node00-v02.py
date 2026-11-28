"""Calibrate coupled sensors and recover the downstream coordinates."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _linear_fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _shape(x, name, k):
    if name == 'sin':
        return np.sin(k*x)/k-x
    if name == 'tanh':
        return np.tanh(k*x)/k-x
    if name == 'atan':
        return np.arctan(k*x)/k-x
    if name == 'rat':
        return x/(1+k*x*x)-x
    return x**3


def _shape_derivative(x, name, k):
    if name == 'sin':
        return np.cos(k*x)-1
    if name == 'tanh':
        return -np.tanh(k*x)**2
    if name == 'atan':
        return 1/(1+(k*x)**2)-1
    if name == 'rat':
        return (1-k*x*x)/(1+k*x*x)**2-1
    return 3*x*x


def _fit_sensor(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.max(np.abs(u-v), axis=1) < 2
    u, v = u[keep], v[keep]
    models = []
    for j in range(4):
        a, b, y = u[:,j], u[:,(j+2)%4], v[:,j]
        best = None
        for f in ('sin','tanh','atan','rat','cube'):
            for g in ('sin','tanh','atan','rat','cube'):
                def design(k):
                    return np.column_stack((np.ones(len(a)), a, b, _shape(a,f,k[0]), _shape(b,g,k[1])))
                def residual(k):
                    d = design(k)
                    return d @ _linear_fit(d,y)-y
                k = least_squares(residual, [1.3, 2.], bounds=(.08, 4.5), max_nfev=75).x
                d = design(k)
                co = _linear_fit(d,y)
                r = d@co-y
                n = len(a)
                score = n*np.log(max(np.mean(r*r),1e-30))+(5+(f!='cube')+(g!='cube'))*np.log(n)
                if best is None or score < best[0]:
                    best = (score, f, g, k, co)
        models.append(best[1:])
    return models


def _undo_sensor(x, models):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for j in range(2):
        cols = [j,j+2]
        pair = [models[i] for i in cols]
        for p, obs in enumerate(x[:,cols]):
            def evaluate(q, jac=False):
                z = np.empty((2,2) if jac else (2,))
                for i, (f,g,k,co) in enumerate(pair):
                    a,b=q[i],q[1-i]
                    if jac:
                        z[i,i] = co[1]+co[3]*_shape_derivative(a,f,k[0])
                        z[i,1-i] = co[2]+co[4]*_shape_derivative(b,g,k[1])
                    else:
                        z[i] = co[0]+co[1]*a+co[2]*b+co[3]*_shape(a,f,k[0])+co[4]*_shape(b,g,k[1])-obs[i]
                return z
            out[p,cols] = least_squares(evaluate, obs, jac=lambda q:evaluate(q,True), max_nfev=45, gtol=1e-11).x
    return out


def _map_coordinates(panel, x, kind='sin', field='v'):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', field))
    def design(a, k):
        one = np.ones(len(a))
        if kind=='linear': return np.column_stack((one,a))
        if kind=='quad': return np.column_stack((one,a,a*a))
        if kind=='cube': return np.column_stack((one,a,a**3))
        return np.column_stack((one,a,_shape(a,kind,k)))
    k = 1.
    if kind not in ('linear','quad','cube'):
        def loss(k):
            d = design(u,k)
            return np.sum((d@_linear_fit(d,v)-v)**2)
        k = minimize_scalar(loss, bounds=(.12,4.5), method='bounded').x
    return design(x,k) @ _linear_fit(design(u,k),v)



def _basis_bank(z, raw, study):
    """Construct coordinate hypotheses from calibration panels and their interactions."""
    B = _map_coordinates(study['panel_b'], z, 'sin')
    C = _map_coordinates(study['panel_c'], z, 'atan', 'v1')
    D = _map_coordinates(study['panel_c'], z, 'quad', 'v2')
    features = [z, B, C, D]
    for v in (B,C,D):
        features += [np.column_stack([v[:,i]*v[:,j] for i in range(4) for j in range(i,4)])]
    for a,b in ((B,C),(B,D),(C,D),(z,B),(z,C),(z,D)):
        features += [a*b, (a[:,:,None]*b[:,None,:]).reshape(len(a),16)]
        for k in (.5,1.,2.,4.):
            features += [a/(1+k*b*b), b/(1+k*a*a), a*np.tanh(k*b), b*np.tanh(k*a), np.tanh(k*(a+b)), np.tanh(k*(a-b))]
    for k in (.5, 1., 1.5, 2., 2.4, 2.8, 3.2, 4., 5.):
        features += [np.sin(k*z),np.tanh(k*z),np.arctan(k*z),np.cos(k*z)]
    from itertools import combinations_with_replacement
    polynomial = []
    for degree in (2,3,4):
        polynomial.append(np.column_stack([np.prod(z[:,ix],axis=1) for ix in combinations_with_replacement(range(4),degree)]))
    composed=[]
    for u in (raw,B,C,D):
        composed += [_map_coordinates(study['panel_b'],u,'sin'),_map_coordinates(study['panel_c'],u,'atan','v1'),_map_coordinates(study['panel_c'],u,'quad','v2')]
    return [z, B,C,D],polynomial,features,composed


def _fit_indices(z,raw,study,n):
    """Estimate two trajectory indices and a scalar index with training-only model selection."""
    trajectory=np.asarray(study['train']['traj'],float)
    _,_,modes=np.linalg.svd(trajectory,full_matrices=False)
    y=np.asarray(study['train']['y'],float).reshape(-1)
    target=np.column_stack((trajectory@modes[:2].T,y))
    center=target.mean(axis=0)
    target=target-center
    basic,poly,features,composed=_basis_bank(z,raw,study)
    opts=[]
    def standardized(parts):
        a=np.column_stack(parts)
        a=a-a[:n].mean(axis=0)
        return a/np.maximum(a[:n].std(axis=0),1e-8)
    def evaluate(kernel):
        K=kernel[:n,:n]
        mu=K.mean(0); m=mu.mean()
        K=(K-mu[:,None]-mu[None,:]+m)
        cross=kernel[:,:n]-kernel[:,:n].mean(1,keepdims=True)-mu+m
        eig,Q=np.linalg.eigh(K)
        eig=np.maximum(eig,0.)
        QT=Q.T@target
        for a in (1e-5,1e-4,1e-3,.01,.05,.2,1.,4.):
            inv=1/(eig+a)
            dual=Q@(QT*inv[:,None]); denom=(Q*Q)@inv-1/(n*a)
            err=dual/np.maximum(denom[:,None],1e-7)
            opts.append((np.mean(abs(err),0),cross@dual))
    for parts in (basic[:1]+poly[:1],basic[:1]+poly[:2],basic[:1]+poly,basic,features[:4]+features[4:14],features,features+poly,features+poly+composed):
        X=standardized(parts); evaluate(X@X.T/X.shape[1])
    for points in (basic[0],np.column_stack(basic),np.column_stack(basic[1:])):
        X=standardized([points]); gram=X@X.T/X.shape[1]; sq=np.diag(gram); d=np.maximum(sq[:,None]+sq-2*gram,0)
        for scale in (.07,.2,.5,1.,2.):
            evaluate(np.exp(-scale*d)); r=np.sqrt(3*scale*d); evaluate((1+r)*np.exp(-r))
    # Sparse forward selection complements the smooth kernels for sharply structured coordinates.
    for parts in (features[:14]+poly,features+poly,features+poly+composed):
        X=standardized(parts); A=X[:n]; N=X.shape[1]; J=np.ones((n,1))/np.sqrt(n)
        for mode in (0,1,2,3):
            Q=J.copy(); R=target.copy(); Ares=A.copy(); selected=[]
            for step in range(min(18,n//3)):
                norm=np.linalg.norm(Ares,axis=0); norm[selected]=np.inf
                cor=(Ares.T@R)/np.maximum(norm[:,None],1e-5)
                merit=np.abs(cor[:,mode]) if mode<3 else np.linalg.norm(cor/np.maximum(target.std(axis=0),.015),axis=1)
                j=int(np.argmax(merit)); q=Ares[:,j]/max(norm[j],1e-12)
                if norm[j]<1e-5 or not np.isfinite(norm[j]):break
                selected.append(j); Q=np.column_stack((Q,q)); R-=np.outer(q,q@R); Ares-=np.outer(q,q@Ares)
                err=R/np.maximum(1-(Q*Q).sum(1,keepdims=True),.01)
                # Discount the optimism from searching a large feature dictionary.
                penalty=1+(.08+.015*step)*np.log2(N)/8
                score=np.mean(abs(err),0)*penalty
                if mode<3: score[np.arange(3)!=mode]*=1.20
                opts.append((score,X[:,selected]@np.linalg.lstsq(A[:,selected],target,rcond=None)[0]))
    rates=np.stack([e for e,_ in opts]); predictions=np.stack([p for _,p in opts]); out=[]
    for j in range(3):
        order=np.argsort(rates[:,j]); best=order[:min(8,len(order))]; scale=max(rates[best[0],j],1e-7)
        w=np.exp(-18*(rates[best,j]/scale-1)); out.append(w@predictions[best,:,j]/w.sum()+center[j])
    indices=np.column_stack(out)
    # The orthogonal residual reference direction completes the coordinate chart.
    anchor=basic[1][:,0]; des=np.column_stack((np.ones(len(z)),indices)); anchor=anchor-des@np.linalg.lstsq(des[:n],anchor[:n],rcond=None)[0]
    return np.column_stack((indices[:,:2],indices[:,2],.1*anchor))


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']['x']; test=study['test']['x']
    x=np.concatenate((train,test)); z=_undo_sensor(x,_fit_sensor(study['panel_a']))
    coords=_fit_indices(z,x,study,len(train))
    return {'c':coords[:len(train)],'query':coords[len(train):]}
