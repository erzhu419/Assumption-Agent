"""Recover latent coordinates from the paired calibration panels."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _a_basis(z):
    w = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)


def _sensor_inverse(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u','v'))
    a = _a_basis(u).transpose(1,0,2)
    rng = np.random.default_rng(2)
    ids = np.argsort(rng.random((96,len(u))),axis=1)[:,:6]
    coeff = np.linalg.pinv(a[:,ids]) @ v.T[:,ids,None]
    err = np.sqrt(np.sum((np.einsum('dnp,dsp->dsn',a,coeff[...,0])-v.T[:,None,:])**2,axis=0))
    count = max(6,int(.8*len(u)))
    loss = np.mean(np.partition(err,count-1,axis=1)[:,:count]**2,axis=1)
    r = err[np.argmin(loss)]
    keep = r <= max(8*np.median(r),1e-3)
    p = np.stack([np.linalg.lstsq(a[j,keep],v[keep,j],rcond=None)[0] for j in range(u.shape[1])])
    def forward(z):
        return np.einsum('dp,dp->d',_a_basis(z),p)
    def jac(z):
        j = np.arange(len(z)); w = np.roll(z,-1)
        out = np.zeros((len(z),len(z)))
        out[j,j] = p[:,1]+p[:,3]*w+3*p[:,4]*z*z
        out[j,(j+1)%len(z)] = p[:,2]+p[:,3]*z
        return out
    return np.array([least_squares(lambda z:forward(z)-row,row,jac=jac,gtol=1e-12).x for row in x]).reshape(x.shape)


def _fit(panel, key, fun, bounds):
    u,v = (np.asarray(panel[k],dtype=float) for k in ('u',key))
    def basis(x,s): return np.column_stack((np.ones(len(x)),x,fun(x,s)))
    def loss(s):
        a=basis(u,s); inv=np.linalg.pinv(a); r=v-a@(inv@v)
        return np.mean(np.abs(r/np.maximum(1-np.sum(a*inv.T,axis=1),1e-8)[:,None]))
    s=minimize_scalar(loss,bounds=bounds,method='bounded').x
    p=np.linalg.lstsq(basis(u,s),v,rcond=None)[0]
    return lambda x:basis(x,s)@p


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; x=np.asarray(study['train']['x'],dtype=float)
    z=_sensor_inverse(study['panel_a'],np.concatenate((x,study['test']['x'])))
    b=_fit(study['panel_b'],'v',lambda q,s:q**3/(1+s*q*q),(.01,2.))
    d=_fit(study['panel_c'],'v2',lambda q,s:np.expm1(-s*q*q),(.02,1.))
    h=_fit(study['panel_c'],'v1',lambda q,s:np.arctan(s*q),(.1,5.))
    c=_coordinates_from_calibration(z,(b,h,d),study['train'],len(x))
    return {'c':c[:len(x)],'query':c[len(x):]}


def _coordinates_from_calibration(z, panels, train, n):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    b, h, d = panels
    p, q, r = b(z), h(z), d(z)
    bases = [z, p, q, r, b(p), h(p), d(p), b(q), h(q), d(q), b(r), h(r), d(r)]
    cores = [z, p, q, r]
    blocks = list(bases)
    for i, v in enumerate(cores):
        for w in cores[i:]:
            blocks.extend((v*w, v/(1+w*w), w/(1+v*v)))
            prod = (v[:,:,None]*w[:,None,:]).reshape(len(z),-1)
            blocks.append(prod)
    for v in cores:
        blocks.extend((v**2, v**3, np.tanh(2*v), np.arctan(3*v)))
        ip = v @ v.T
        dist = np.maximum(np.diag(ip)[:,None]+np.diag(ip)[None,:]-2*ip, 0.)
        for rate in (0.4, 1.6, 5.):
            blocks.append(np.exp(-rate*dist))
    for v in (p+q, p+r, q+r, p-q, p-r, q-r, p*q, p*r, q*r, p+q*r, (p+q)/(1+r*r)):
        blocks.extend((b(v),h(v),d(v)))
    grams = []
    for x in blocks:
        if x.shape == (len(z),len(z)):
            m=x[:n,:n].mean(); grams.append(x/max(m,0.1))
        else:
            x=x-x[:n].mean(axis=0)
            x=x/np.maximum(x[:n].std(axis=0),1e-7)
            grams.append((x@x.T)/x.shape[1])
    k=np.asarray(grams)
    kt=np.ascontiguousarray(k[:,:n,:n]).reshape(len(k),-1)
    tr=np.asarray(train['traj'],float)
    mean=tr.mean(0)
    _,_,vt=np.linalg.svd(tr-mean,full_matrices=False)
    y=np.column_stack((np.asarray(train['y'],float), (tr-mean)@vt[:2].T))
    mu=y.mean(0); sd=np.maximum(y.std(0),1e-8); y=(y-mu)/sd
    eye=np.eye(n)
    results=[]
    for j in range(3):
        target=y[:,j]
        def evaluate(logw, predict=False):
            w=np.exp(logw)
            cov=(w[:-1]@kt).reshape(n,n)+(1e-9+w[-1])*eye+0.3
            fact=cho_factor(cov,check_finite=False)
            alpha=cho_solve(fact,target,check_finite=False)
            if predict:
                cross=np.einsum('i,ijk->jk',w[:-1],k[:,:,:n])+0.3
                return cross@alpha
            ic=cho_solve(fact,eye,check_finite=False)
            change=ic-np.outer(alpha,alpha)
            # A weak weight penalty discourages unnecessary large components.
            obj=0.5*(target@alpha)+np.sum(np.log(np.diag(fact[0])))+0.04*w[:-1].sum()
            grad=0.5*np.r_[kt@change.ravel(),np.trace(change)]
            grad[:-1]+=0.04
            return obj, grad*w
        init=np.r_[np.full(len(k),-4.),-2.5]
        fit=minimize(evaluate,init,jac=True,method='L-BFGS-B',bounds=[(-14.,3.)]*len(k)+[(-9.,1.)],options={'maxiter':165,'ftol':1e-8})
        results.append(evaluate(fit.x,True))
    c=np.column_stack((*results,np.zeros(len(z))))
    return c
