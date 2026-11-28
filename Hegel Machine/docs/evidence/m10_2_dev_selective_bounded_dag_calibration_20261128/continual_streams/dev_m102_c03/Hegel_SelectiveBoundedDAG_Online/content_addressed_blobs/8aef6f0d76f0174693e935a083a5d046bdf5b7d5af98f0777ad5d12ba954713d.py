import numpy as np
from itertools import combinations


def _features(z):
    s = [np.ones(len(z)), z]
    for j in range(4):
        u = z[:, j:j+1]
        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])
    groups = []
    for i, j in combinations(range(4), 2):
        u, v = z[:, i:i+1], z[:, j:j+1]
        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]
        for sign in (-1, 1):
            w = (u+sign*v)/1.4
            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])
        groups.append(np.column_stack(g))
    return s, groups


def _baseline(inputs: dict) -> dict:
    study = inputs['__input__']
    tr = study['train']
    x = np.asarray(tr['x'])
    xt = np.asarray(study['test']['x'])
    n = len(x)
    z = (np.vstack([x, xt])-x.mean(0))/np.maximum(x.std(0), 1e-12)
    q = -np.expm1(-np.asarray(study['eval_times'])[:, None]*np.array([.392, 1.857]))
    a = np.linalg.lstsq(q, tr['traj'].T, rcond=None)[0].T
    s, g = _features(z)
    f = np.column_stack(s+g+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])[:, 1:]
    ys = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)
    fs = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)

    def ols(ids, y):
        mat = np.column_stack([np.ones(len(z)), f[:, ids]])
        return mat[n:] @ np.linalg.lstsq(mat[:n], y, rcond=None)[0]

    m = np.column_stack(s+[g[j] for j in (0, 1, 3)])
    m = m/np.maximum(np.linalg.norm(m[:n], axis=0), 1e-8)
    slow = m[n:] @ m[:n].T @ np.linalg.solve(m[:n] @ m[:n].T+.02*np.eye(n), a[:, 0])
    y = ols(ys, tr['y'])
    pred = np.column_stack([slow, ols(fs, a[:, 1])]) @ q.T
    return {'y_hat': y, 'state_pred': pred}

from scipy.optimize import minimize, least_squares
from scipy.linalg import cho_factor, cho_solve


def _linear(z):
    return np.column_stack((np.ones(len(z)),z))


def _calibrate(study,x):
    a=study['panel_a']; u,v=np.asarray(a['u']),np.asarray(a['v']); f=_linear(u)
    k=np.linalg.norm(v-u,axis=1)<2
    if k.sum()<8:k=np.ones(len(u),dtype=bool)
    for _ in range(4):
        b=np.linalg.lstsq(f[k],v[k],rcond=None)[0]
        e=np.linalg.norm(f@b-v,axis=1)
        kk=e<max(.003,3*np.median(e))
        if kk.sum()>=8:k=kk
    return np.linalg.solve(b[1:].T,(x-b[0]).T).T


def _dictionary(z,study):
    u=np.asarray(study['panel_c']['u']); v=study['panel_c']['v2']
    def loss(p):
        f=_linear(np.sin(p[0]*u)**2);return (f@np.linalg.lstsq(f,v,rcond=None)[0]-v).ravel()
    freq=least_squares(loss,[1.2],bounds=(.65,1.65)).x[0]
    columns=[]
    def add(f):columns.append(np.column_stack(f))
    add([z]);add([np.sin(freq*z)**2]);add([np.sin(freq*z)]);add([z/(1+.45*abs(z))**2]);add([np.tanh(1.2*z)]);add([z*z]);
    for j in range(4):
        x=z[:,j]; add([x,np.sin(freq*x), np.tanh(1.2*x),x/(1+.45*abs(x))**2, np.arctan(1.8*x)]);add([x*x,np.sin(freq*x)**2,np.tanh(1.2*x)**2,np.exp(-x*x),x**4]);
    for i,j in combinations(range(4),2):
        x,y=z[:,i],z[:,j]; sx,sy=np.sin(freq*x),np.sin(freq*y); tx,ty=np.tanh(1.2*x),np.tanh(1.2*y); rx,ry=x/(1+.45*abs(x))**2,y/(1+.45*abs(y))**2
        add([x*y,sx*sy,tx*ty,rx*ry,x*y/(1+.5*(x*x+y*y)),np.cos(x+y),np.cos(x-y)])
        add([sx*sx*sy,sx*sy*sy,tx*tx*ty,tx*ty*ty,x*x*y,x*y*y,rx*rx*ry,rx*ry*ry,np.sin(x+y),np.sin(x-y),np.tanh(x+y),np.tanh(x-y)])
        add([x*x*y*y,sx*sx*sy*sy,tx*tx*ty*ty,sx**3*sy,sx*sy**3,tx**3*ty,tx*ty**3]);
    for j in combinations(range(4),3):
        p=z[:,j]; add([np.prod(f,axis=1) for f in (p,np.sin(freq*p),np.tanh(1.2*p),p/(1+.45*abs(p))**2)])
    return columns


def _readout(z,study,Y):
    n=len(Y); fs=_dictionary(z,study); K=[]; L=[]
    for f in fs:
        f=f-f[:n].mean(0); f=f/np.maximum(np.sqrt(np.mean(f[:n]**2,axis=0)),1e-8)/np.sqrt(f.shape[1]); K.append(f[:n]@f[:n].T);L.append(f[n:]@f[:n].T)
    K=np.asarray(K);L=np.asarray(L); k=K.reshape(len(K),-1); l=L.reshape(len(K),-1); ident=np.eye(n)
    out=[]
    for col in Y.T:
        ym=col.mean(); sd=max(col.std(),1e-8); y=(col-ym)/sd
        def evaluate(p, pred=False):
            w=np.exp(p); c=cho_factor((w[:-1]@k).reshape(n,n)+w[-1]*ident,check_finite=False); a=cho_solve(c,y,check_finite=False)
            if pred:return (w[:-1]@l).reshape(-1,n)@a*sd+ym
            d=cho_solve(c,ident,check_finite=False)-np.outer(a,a)
            return .5*y@a+np.log(np.diag(c[0])).sum(), .5*w*np.r_[k@d.ravel(),np.trace(d)]
        p=minimize(evaluate,np.r_[np.full(len(K),-4.),-6.],jac=True,method='L-BFGS-B',bounds=[(-17,6)]*len(K)+[(-11,-1)],options={'maxiter':160,'ftol':1e-8}).x
        out.append(evaluate(p,True))
    return np.column_stack(out)


def _infer(study):
    tr=study['train']; n=len(tr['x']); z=_calibrate(study,np.vstack((tr['x'],study['test']['x']))); t=np.asarray(study['eval_times'])
    traj=np.asarray(tr['traj']).T
    def q(r):return -np.expm1(-t[:,None]*r)
    def res(r):
        m=q(r); return (m@np.linalg.lstsq(m,traj,rcond=None)[0]-traj).ravel()
    m=q(least_squares(res,[.41,1.81],bounds=([.12,.9],[.85,3.5])).x)
    Y=np.column_stack((tr['y'],np.linalg.lstsq(m,traj,rcond=None)[0].T)); p=_readout(z,study,Y)
    return p[:,0],p[:,1:]@m.T


def _scalar_kernels(z, study):
    # Recover the two panel transfer laws independently of the response labels.
    p=study['panel_c']; u=np.asarray(p['u']); v=p['v1']; w=p['v2']
    def rfun(x,p):return x/(1+p[0]*abs(x)+p[1]*x*x+p[2]*abs(x)**3)
    def residual(f,y):return (f@np.linalg.lstsq(f,y,rcond=None)[0]-y).ravel()
    par=least_squares(lambda p:residual(_linear(rfun(u,p)),v),[.89,.30,-.033],bounds=([.6,.12,-.06],[1.2,.46,.03])).x
    fr=least_squares(lambda p:residual(_linear(np.sin(p[0]*u)**2),w),[1.2],bounds=(.85,1.5)).x[0]
    return [z,np.sin(fr*z),rfun(z,par),np.tanh(z),np.tanh(1.35*z),np.arctan(1.8*z)]


def _scalar_models(z,study,y,configs):
    from itertools import combinations_with_replacement
    from scipy.spatial.distance import cdist
    n=len(y); mu=y.mean(); sd=max(y.std(),1e-9); yy=(y-mu)/sd; eye=np.eye(n)
    coords=_scalar_kernels(z,study); polys={}; dists={}; features={}
    def gram(x):
        x=x-x[:n].mean(0); x=x/np.maximum(np.sqrt(np.mean(x[:n]**2,axis=0)),1e-9)/np.sqrt(x.shape[1]); return x@x[:n].T
    for k,x in enumerate(coords):
        x=x/np.sqrt(np.mean(x[:n]**2,axis=0)); dists[k]=cdist(x,x[:n],metric='sqeuclidean'); P={}
        for deg in range(1,6):
            by_support={}
            for j in combinations_with_replacement(range(4),deg):by_support.setdefault(tuple(sorted(set(j))),[]).append(np.prod(x[:,j],axis=1))
            P[deg]={j:gram(np.column_stack(f)) for j,f in by_support.items()}
        polys[k]=P
    def summed(k,deg,power):
        return sum(power**(d-1)*sum(polys[k][d].values())/len(polys[k][d]) for d in range(1,deg+1))
    def fit(K,reg):
        C=cho_factor(K[:n]+reg*eye,check_finite=False); A=cho_solve(C,eye,check_finite=False); a=A@yy
        return sd*np.sqrt(np.mean((a/A.diagonal())**2)),mu+sd*K[n:]@a
    def optimize(K,lo,criterion):
        K=np.asarray(K); m=len(K); F=K[:,:n].reshape(m,-1).copy(); G=K[:,n:].reshape(m,-1).copy()
        def fun(p,predict=False):
            w=np.exp(p); C=cho_factor((w[:-1]@F).reshape(n,n)+w[-1]*eye,check_finite=False); A=cho_solve(C,eye,check_finite=False); a=A@yy
            err=a/A.diagonal(); score=sd*np.sqrt(np.mean(err**2))
            if predict:return score,mu+sd*(w[:-1]@G).reshape(-1,n)@a
            if criterion=='ml':
                D=.5*(A-np.outer(a,a)); loss=.5*yy@a+np.log(np.diag(C[0])).sum()
            else:
                d=A.diagonal(); ax=A@(err/d); D=(-2*np.outer(ax,a)+2*(A*(err**2/d)[None,:])@A)/n; loss=np.mean(err**2)
            return loss,w*np.r_[F@D.ravel(),np.trace(D)]
        p=minimize(fun,np.r_[np.full(m,-3.),-6.],jac=True,method='L-BFGS-B',bounds=[(-15,6)]*m+[(lo,-1)],options={'maxiter':90,'ftol':2e-7}).x
        return fun(p,True)
    cache={}; res=[]
    for typ,ks,d,g,reg in configs:
        if typ in ('poly','rbf','matern','plus'):
            key=typ,ks,d,g
            if key not in cache:
                if typ=='poly':cache[key]=sum(summed(k,d,g) for k in ks)/len(ks)
                elif typ=='rbf':cache[key]=sum(np.exp(-g*dists[k]) for k in ks)/len(ks)
                elif typ=='matern':cache[key]=sum((1+np.sqrt(g*dists[k])+g*dists[k]/3)*np.exp(-np.sqrt(g*dists[k])) for k in ks)/len(ks)
                else:cache[key]=sum(summed(k,d,.6)+np.exp(-g*dists[k]) for k in ks)/len(ks)
            out=fit(cache[key],reg)
        else:
            K=[]
            if typ in ('tensor','loo'):
                for k in ks:
                    for j in sorted(set.union(*(set(polys[k][q]) for q in range(1,d+1)))):
                        K.append(sum(g**(q-len(j))*polys[k][q][j] for q in range(len(j),d+1) if j in polys[k][q]))
            else:
                for k in ks:
                    K.extend(sum(polys[k][q].values())/len(polys[k][q]) for q in range(1,d+1)); K.extend(np.exp(-h*dists[k]) for h in (.04,.15,.5))
            out=optimize(K,reg,'loo' if typ=='loo' else 'ml')
        res.append(out)
    return res

_SCALAR_CONFIGS = [('evidence', (1, 2), 4, 0.65, -7)]


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; tr=study['train']; y,traj=_infer(study)
    z=_calibrate(study,np.vstack((tr['x'],study['test']['x'])))
    estimates=_scalar_models(z,study,np.asarray(tr['y']),_SCALAR_CONFIGS)
    scalar=np.mean([p for _,p in sorted(estimates,key=lambda a:a[0])[:1]],axis=0)
    y=0.25*y+0.75*scalar
    ref=_baseline({'__input__':study})
    return {'panel_pred':np.column_stack((2*y-ref['y_hat'],(traj-.35*ref['state_pred'])/.65))}
