"""Recover calibrated coordinates and build the frozen heads' feature blocks."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    opp = (np.arange(4) + 2) % 4
    def design(w):
        return np.stack((np.ones_like(u), u, u**3, np.sin(w*u[:,opp])), axis=2)
    w = np.ones(len(u))
    a = design(2.4)
    for _ in range(20):
        c = np.array([_fit(a[:,j]*w[:,None],v[:,j]*w) for j in range(4)])
        r = np.linalg.norm(np.einsum('njk,jk->nj', a,c)-v,axis=1)
        w = np.sqrt(np.minimum(1.,1.5*max(np.median(r),1e-12)/np.maximum(r,1e-12)))
    good = r <= 5*max(np.median(r),1e-12)
    def prof(freq):
        g=design(freq)[good]
        c=np.array([_fit(g[:,j],v[good,j]) for j in range(4)])
        return c,np.einsum('njk,jk->nj',g,c)-v[good]
    f=least_squares(lambda p:prof(p[0])[1].ravel(),[2.4],bounds=(1.5,3.3)).x[0]
    c=prof(f)[0]
    idx=np.arange(4)
    def res(q,row):
        return c[:,0]+c[:,1]*q+c[:,2]*q**3+c[:,3]*np.sin(f*q[opp])-row
    def jac(q, row):
        a=np.diag(c[:,1]+3*c[:,2]*q*q)
        a[idx,opp]=f*c[:,3]*np.cos(f*q[opp]);return a
    return np.array([least_squares(res,np.clip((row-c[:,0])/c[:,1],-1.79,1.79),jac=jac,args=(row,),bounds=(-1.8,1.8),gtol=1e-11).x for row in x])


def _affine(x):
    return np.column_stack((np.ones(len(x)),x))


def _poly(x):
    i,j=np.triu_indices(4)
    return np.column_stack((_affine(x), x[:,i]*x[:,j]))


def _panel(panel, k, x):
    u, v = (np.asarray(panel[j], float) for j in ('u', k))
    if k == 'v':
        start, lo, hi = [2.9, -.4], [2., -.8], [4., -.05]
        def warp(z, p): return z + p[1]*np.sin(p[0]*z)
    elif k == 'v1':
        start, lo, hi = [2.4, 6.], [.8, 0.], [5., 20.]
        def warp(z, p): return z + p[1]*np.arctan(p[0]*z)
    else:
        start, lo, hi = [1.3, -.65], [.5, -1.5], [2.5, -.05]
        def warp(z, p): return z + p[1]*(np.cos(p[0]*z)-1.)
    def residual(p):
        a = _affine(warp(u, p))
        return (a @ _fit(a, v) - v).ravel()
    p = least_squares(residual, start, bounds=(lo, hi)).x
    return _affine(warp(x, p)) @ _fit(_affine(warp(u, p)), v)


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    train,test=(np.asarray(study[k]['x'],float) for k in ('train','test'))
    x=_calibrate(study['panel_a'],np.vstack((train,test)))
    b=_panel(study['panel_b'],'v',x)
    c=_panel(study['panel_c'],'v1',x)
    d=_panel(study['panel_c'],'v2',x)
    i,j=np.triu_indices(4)
    r=np.column_stack((b,b[:,i]*b[:,j],np.ones(len(x)),x,_poly(x)[:,5:],c,d))
    z=np.column_stack((c,np.ones(len(x)),x,_poly(x)[:,5:],d))
    n=len(train)
    return {'readout_train':r[:n], 'readout_test':r[n:], 'z':z[:n], 'zt':z[n:]}


def _latent(panel, key, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', key))
    # Profile the affine panel mixing; use the unmixed coordinates for kernels.
    if key == 'v':
        def base(z,p): return z+p[1]*np.sin(p[0]*z)
        start,lo,hi=[2.9,-.4],[2.,-.8],[4.,-.05]
    elif key == 'v1':
        def base(z,p): return z+p[1]*np.arctan(p[0]*z)
        start,lo,hi=[2.4,6.],[.8,0.],[5.,20.]
    else:
        def base(z,p): return z+p[1]*(np.cos(p[0]*z)-1.)
        start,lo,hi=[1.3,-.65],[.5,-1.5],[2.5,-.05]
    def residual(p, f):
        A=_affine(f(u,p));return (A@_fit(A,v)-v).ravel()
    p=least_squares(residual,start,args=(base,),bounds=(lo,hi)).x
    warp=base
    if key!='v2':
        def augmented(z,p): return base(z,p)+p[2]*z**3/(1+p[3]*z*z)
        opt=min((least_squares(residual,[*p,0.,k],args=(augmented,),bounds=([*lo,-12.,.01],[*hi,12.,30.]),max_nfev=120) for k in (.4,3.,12.)),key=lambda o:o.cost)
        if np.linalg.norm(opt.fun)<.75*np.linalg.norm(residual(p,base)):
            p,warp=opt.x,augmented
    return warp(x,p),_affine(warp(x,p))@_fit(_affine(warp(u,p)),v)


def _kernel_features(maps, n):
    """Build separate covariance components for main effects and interactions."""
    x,b,c,d,B,C,D=maps
    def std(z):return (z-z[:n].mean(0))/np.maximum(z[:n].std(0),1.e-8)
    xs=[std(z) for z in maps]; x,b,c,d,B,C,D=xs
    N=len(x); out=[np.ones((N,N))]
    def add(A):
        scale=np.trace(A[:n,:n])/n
        if scale>1.e-12: out.append(A/scale)
    def gram(A):return A@A.T
    def distances(A):return (A[:,None,:]-A[None,:,:])**2
    for Z in xs:
        G=gram(Z)/4
        add(G);add(G**2)
        for i in range(4):
            add(gram(Z[:,i:i+1]));add(gram(Z[:,i:i+1]**2))
        R=distances(Z)
        for L in (.5,2.,8.,32.):
            A=np.exp(-R/L)
            add(np.mean(A,axis=2))
            add((A.sum(2)**2-(A**2).sum(2))/12)
            add(np.prod(A,axis=2))
            if L in (2.,8.):
                for i in range(4): add(A[:,:,i])
    for i,Z in enumerate(xs):
        for W in xs[:i]: add(gram(Z)*gram(W))
    for W in (b,c,d):
        R=(distances(W)+.15*distances(x))/1.15
        for L in (2.,8.):
            A=np.exp(-R/L)
            for i in range(4):
                for j in range(i): add(A[:,:,i]*A[:,:,j])
    return np.asarray(out)


def _profile_features(K, y, n):
    from scipy.linalg import cho_factor, cho_solve, eigh, hadamard
    from scipy.optimize import minimize
    y=np.asarray(y,float).reshape(n,-1)
    # Fit covariance hyperparameters only from the current training partition.
    u,ss,_=np.linalg.svd(y-y.mean(0),full_matrices=False)
    m=min(y.shape[1],3)
    if m==1:
        Y=(y-y.mean(0))/max(float(y.std()),1.e-8)
    else:
        Y=u[:,:m]*(ss[:m]/max(ss[0],1.e-8))
        Y=Y*np.sqrt(n)
    eye=np.eye(n)
    A=np.concatenate((K[:,:n,:n],eye[None])).reshape(-1,n*n)
    ref=np.r_[np.full(len(K),-5.5),-7.];ref[0]=-2.
    def fun(q):
        w=np.exp(q); F=cho_factor((w@A).reshape(n,n),check_finite=False)
        v=cho_solve(F,Y,check_finite=False)
        d=q-ref; Q=cho_solve(F,eye,check_finite=False)-v@v.T/Y.shape[1]
        return np.log(F[0].diagonal()).sum()+np.sum(v*Y)/(2*Y.shape[1])+.035*(d@d)/2, .5*(A@Q.ravel())*w+.035*d
    opt=minimize(fun,ref,jac=True,method='L-BFGS-B',bounds=[(-16.,5.)]*len(K)+[(-13.,-2.5)],options={'maxiter':210,'ftol':1.e-9})
    M=np.tensordot(np.exp(opt.x[:-1]),K,axes=1)
    # Nyström factorization against training landmarks only. The orthogonal
    # Hadamard rotation equalizes training column energies for the frozen head.
    lam,U=eigh(M[:n,:n],check_finite=False); good=lam>max(lam[-1]*1.e-10,1.e-12)
    F=M[:,:n]@(U[:,good]/np.sqrt(lam[good]))
    p=1<<(F.shape[1]-1).bit_length()
    return F@hadamard(p,dtype=float)[:F.shape[1]]/np.sqrt(p)


_baseline_run=run_node

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']
    train,test=(np.asarray(study[k]['x'],float) for k in ('train','test'))
    n=len(train)
    x=_calibrate(study['panel_a'],np.vstack((train,test)))
    b,B=_latent(study['panel_b'],'v',x)
    c,C=_latent(study['panel_c'],'v1',x)
    d,D=_latent(study['panel_c'],'v2',x)
    K=_kernel_features((x,b,c,d,B,C,D),n)
    r=_profile_features(K,study['train']['y'],n)
    z=_profile_features(K,study['train']['traj'],n)
    return {'readout_train':r[:n], 'readout_test':r[n:], 'z':z[:n], 'zt':z[n:]}
