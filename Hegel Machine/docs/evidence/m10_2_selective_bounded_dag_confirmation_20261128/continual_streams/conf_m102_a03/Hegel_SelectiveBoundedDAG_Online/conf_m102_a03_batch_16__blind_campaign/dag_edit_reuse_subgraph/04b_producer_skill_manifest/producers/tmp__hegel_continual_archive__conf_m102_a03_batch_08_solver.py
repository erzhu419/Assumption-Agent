"""Calibrated regression with separate readout and temporal response kernels."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import cho_factor, cho_solve


def fit(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-11)[0]


def affine(x):
    return np.c_[np.ones(len(x)),x]


def calibrate(panel,x):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    good=np.linalg.norm(v,axis=1)<5
    def design(u,j):
        a=u[:,j]; b=u[:,(j+1)%4]
        return np.c_[affine(u),a*b,a**3]
    co=np.column_stack([fit(design(u[good],j),v[good,j]) for j in range(4)])
    def forward(z):
        return np.column_stack([design(z,j)@co[:,j] for j in range(4)])
    g=np.linalg.solve(co[1:5].T,(x-co[0]).T).T
    return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(g,x)])




def _panel_b_warp(panel):
    """Profile out the affine mixing to estimate a shared odd coordinate warp."""
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))

    def warp(x, h):
        a,b,c,d = h
        return x+a*np.arctan(b*x)+d*x**3/(1+c*x*x)

    def residual(h):
        A = affine(warp(u,h))
        return (A@fit(A,v)-v).ravel()

    starts = [(-.45,2.15,12.,-3.3),(-.5,2.,8.,-2.),(-1.,2.,1.,.2),(-.6,1.5,2.,0.)]
    opt = min((least_squares(residual, h, bounds=([-8,0,0,-8],[8,10,16,8]), max_nfev=200) for h in starts), key=lambda r:r.cost)
    return lambda x:warp(x,opt.x)


def _standard(x,n):
    return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)


def _components(z,b,n):
    a,x = _standard(z,n),_standard(b,n)
    A,G = (q@q.T/q.shape[1] for q in (a,x))
    R = ((x[:,None]-x[None,:])**2+.25*(a[:,None]-a[None,:])**2)/1.25
    K=[np.ones_like(A),A,A*A,G,G*G,G*A]
    for L in (1.5,6,24):
        S=np.exp(-R/L)
        M=S.sum(2)
        K.extend([M/4,(M*M-(S*S).sum(2))/12])
        K.extend(S[:,:,i] for i in range(4))
        K.extend(S[:,:,i]*S[:,:,j] for i in range(4) for j in range(i))
    return np.array(K)


def _profile_kernel(B,Y,n,penalty=.02):
    Y=np.asarray(Y).reshape(n,-1)
    mu=Y.mean(0)
    sd=np.maximum(Y.std(0),1e-8)
    M=(Y-mu)/sd
    eye=np.eye(n)
    p=len(B)
    A=np.concatenate([B[:,:n,:n],eye[None]],axis=0).reshape(p+1,-1)
    ref=np.r_[np.full(p,-3.),-7.]
    ref[0]=-1

    def objective(q):
        v=np.exp(q)
        F=cho_factor((v@A).reshape(n,n),check_finite=False)
        alpha=cho_solve(F,M,check_finite=False)
        D=(cho_solve(F,eye,check_finite=False)-alpha@alpha.T/M.shape[1]).ravel()
        d=q-ref
        val=np.log(F[0].diagonal()).sum()+np.sum(M*alpha)/(2*M.shape[1])+penalty*(d@d)/2
        return val,(A@D)*v/2+penalty*d

    O=minimize(objective,ref,jac=True,method='L-BFGS-B',bounds=[(-14,5)]*p+[(-12,-2)],options={'maxiter':160,'ftol':1e-7})
    v=np.exp(O.x)
    F=cho_factor((v@A).reshape(n,n),check_finite=False)
    W=(v[:-1]@B[:,n:,:n].reshape(p,-1)).reshape(-1,n)
    return (W@cho_solve(F,M,check_finite=False))*sd+mu


def solve_batch(study):
    tr=study['train']
    n=len(tr['x'])
    z=calibrate(study['panel_a'],np.r_[tr['x'],study['test']['x']])
    b=_panel_b_warp(study['panel_b'])(z)
    B=_components(z,b,n)
    yh=_profile_kernel(B,tr['y'],n)[:,0]
    a=_standard(z,n)
    w=_standard(z-.4*np.sin(2.9*z),n)
    K=(1+(w@w.T+.1*a@a.T)/4.4)**2
    T=np.asarray(tr['traj'],float)
    m=T.mean(0)
    baseline=K[n:,:n]@cho_solve(cho_factor(K[:n,:n]+.001*np.eye(n)),T-m)+m
    P=.5*baseline+.5*_profile_kernel(B,T,n)
    last=P[:,-1]
    return {'y_hat':yh,'state_pred':P,'diagnostics_vec':np.array([yh.mean(),yh.std(),last.mean(),last.std()])}
