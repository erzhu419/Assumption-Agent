"""Reconstruct the frozen spatial kernel's coordinates and fit a joint readout."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


def _kernel(x, z, w):
    d = x-z
    dot = (x*z).sum(-1)
    r = 1.5*np.exp(-(d*d).sum(-1)/6)
    s = 1.5*np.exp(-(d*d*w).sum(-1)/6)
    val = 1+dot+.15*(dot*dot+((x*z)**2).sum(-1))+r+s
    dx = (1+.30*dot[...,None])*z+.30*x*z*z-d*(r[...,None]+s[...,None]*w)/3
    dz = (1+.30*dot[...,None])*x+.30*z*x*x+d*(r[...,None]+s[...,None]*w)/3
    dw = -s[...,None]*d*d*w/6
    return val, dx, dz, dw


def _coordinates(K, Kq, dim):
    n = len(K)
    evals, U = np.linalg.eigh(K-4)
    x0 = U[:, -dim:]*np.sqrt(np.maximum(evals[-dim:],1e-8))*.80
    i,j = np.tril_indices(n)
    r = np.arange(len(i))[:,None]
    c = np.arange(dim)
    ii,jj = i[:,None]*dim+c,j[:,None]*dim+c
    def residual(p, jac=False):
        x = p[:-dim].reshape(n,dim)
        v, a, b, h = _kernel(x[i],x[j],np.exp(np.clip(p[-dim:],-8,8)))
        if not jac:
            return v-K[i,j]
        J = np.zeros((len(i),(n+1)*dim))
        J[r,ii] = a
        J[r,jj] += b
        J[:,-dim:] = h
        return J
    rng = np.random.default_rng(29)
    best = None
    for k in range(5):
        rot = np.eye(dim) if k == 0 else np.linalg.qr(rng.normal(size=(dim,dim)))[0]
        o = least_squares(residual,np.r_[(x0@rot).ravel(),np.zeros(dim)],jac=lambda p:residual(p,True),method='lm',max_nfev=140,ftol=1e-10)
        if best is None or o.cost < best.cost:
            best = o
        if o.cost < 1e-17:
            break
    p=best.x
    x=p[:-dim].reshape(n,dim)
    w=np.exp(np.clip(p[-dim:],-8,8))
    rows=[]
    for kq in Kq:
        seed=x[np.argmax(kq/np.sqrt(K.diagonal()))]
        f=lambda z:_kernel(z[None,:],x,w)[0]-kq
        jac=lambda z:_kernel(z[None,:],x,w)[1]
        ans=least_squares(f,seed,jac=jac,method='lm',max_nfev=65)
        rows.append(ans.x)
    return x, np.asarray(rows).reshape(-1,dim)


def _predict(x, z, a, b, y):
    u,v=np.c_[x,a],np.c_[z,b]
    center=u.mean(0)
    scale=np.maximum(u.std(0),1e-8)
    u,v=(u-center)/scale,(v-center)/scale
    n,d=u.shape
    I=np.eye(n)
    dx=u[:,None]-u
    dz=v[:,None]-u
    i,j=np.tril_indices(d)
    diag=i==j
    h=len(i)
    def cov(p,u,v,D,S,train=False):
        L=np.zeros((d,d)); L[i,j]=np.where(diag,np.exp(p[:h]),p[:h]); w=np.exp(p[h:]); T=D@L
        K=w[0]*np.exp(-(T*T).sum(-1)); P=w[1]*(1+u@v.T/d)**2; R=w[2]*S
        C=K+P+R+w[3]
        if not train:
            return C
        J=np.empty((len(p),n,n))
        J[:h]=(-2*K[:,:,None]*D[:,:,i]*T[:,:,j]*np.where(diag,L[i,j],1)).transpose(2,0,1)
        J[h:]=[K,P,R,np.full_like(K,w[3]),w[4]*I]
        return C+w[4]*I,J
    S=a@a.T
    def loss(p):
        H,J=cov(p,u,u,dx,S,True)
        F=cho_factor(H)
        c=cho_solve(F,y)
        E=cho_solve(F,I)-np.outer(c,c)
        penalty=.05*(p[:h]@p[:h])
        return .5*y@c+np.log(np.diag(F[0])).sum()+penalty,.5*np.einsum('ij,kij->k',E,J)+np.r_[.1*p[:h],np.zeros(5)]
    rng=np.random.default_rng(0)
    best=None
    for k in range(6):
        ini=np.r_[np.where(diag,-1.5,0)+rng.normal(0,.25*k/5,h),0,-3,-1,-1,-9]
        o=minimize(loss,ini,jac=True,method='L-BFGS-B',bounds=[(-5,1) if t else (-3,3) for t in diag]+[(-16,5)]*4+[(-14,-3)],options={'maxiter':180})
        if best is None or o.fun<best.fun:
            best=o
    p=best.x
    H,_=cov(p,u,u,dx,S,True)
    return cov(p,v,u,dz,b@a.T)@cho_solve(cho_factor(H),y)


def run_node(inputs: dict) -> dict:
    p=inputs['node_00']
    state=np.asarray(inputs['node_02']['state_pred'])
    x,q=_coordinates(p['K'],p['Kq'],p['sy'].shape[1])
    yh=_predict(x,q,p['scores'],state@p['modes'].T,p['y_centered']+p['y_mean'])
    return {'y_hat':yh}
