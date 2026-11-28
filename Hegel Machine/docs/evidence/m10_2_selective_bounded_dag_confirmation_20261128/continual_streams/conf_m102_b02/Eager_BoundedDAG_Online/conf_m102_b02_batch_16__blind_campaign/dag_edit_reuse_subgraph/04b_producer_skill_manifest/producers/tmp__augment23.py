
# Penalized kernel heads with training-only leave-one-out stacking.
def _stack23(z,B,C,E,Y):
    from scipy.optimize import minimize
    n=len(Y); N=len(z); I=np.eye(n); b,c,e=B(z),C(z),E(z)
    mu=Y.mean(0); sd=np.maximum(Y.std(0),1e-8); T=(Y-mu)/sd
    coords=[z,b,c,e,np.c_[z,e],np.c_[z,b],np.c_[b,E(b-.5*z)],np.c_[C(b),E(b-z)],np.c_[z,E(b+z)],np.c_[z,b,c,e],np.c_[b,C(b),E(b)],np.c_[z,C(b),E(b)],np.c_[z, B(z+c), C(b+z), E(b+z)]]
    gram=[]
    for x in coords:
        for scheme in (0,1):
            X=x-x[:n].mean(0)
            if scheme: X=X/np.sqrt(np.maximum(X[:n].std(0),.04))
            G=X@X.T; G/=max(G.diagonal()[:n].mean(),1e-8)
            d=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)
            gram.extend([(1+.35*G)**2,(1+.35*G)**3,(1+.35*G)**4,(1+.35*G)**2+.3*np.exp(-.3*d)])
    loo=[]; val=[]
    for G in gram:
        for ridge in (1e-6,3e-5,.001,.025):
            inv=solve(G[:n,:n]+ridge*I,I,assume_a='pos'); a=inv@T
            loo.append(T-a/inv.diagonal()[:,None]); val.append(G[:,:n]@a)
    L=np.array(loo); V=np.array(val); out=[]
    for j in range(T.shape[1]):
        U=L[:,:,j].T; t=T[:,j]; errs=np.mean(np.abs(U-t[:,None]),axis=0); ix=np.argsort(errs)[:16]; U=U[:,ix]; m=len(ix)
        def objective(w):
            err=U@w-t; q=np.sqrt(err*err+.04**2)
            return q.mean()+.02*np.dot(w,w), U.T@(err/q)/n+.04*w
        res=minimize(objective,np.ones(m)/m,method='SLSQP',jac=True,bounds=[(0,1)]*m,constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)}],options={'ftol':1e-9,'maxiter':80})
        out.append(res.x@V[ix,:,j])
    return np.column_stack(out)*sd+mu


def _predict23(s):
    z,B,C,E=coordinates(s); tr=s['train']; times=s['eval_times']; T=tr['traj'].T
    def basis(q):return -np.expm1(-times[:,None]*q)
    def err(q):
        f=basis(q);return (f@fit(f,T)-T).ravel()
    f=basis(least_squares(err,[1.8,.42],bounds=([1.,.15],[3.,.85])).x); modes=fit(f,T).T
    U,sv,R=np.linalg.svd(f,full_matrices=False); rot=R.T; target=np.c_[tr['y'],modes@rot]
    _,K=covs(banks(z,B,C,E),len(T.T)); p=regress(K,target); h=_stack23(z,B,C,E,target)
    b=B(z); x=np.c_[z,b]; n=len(modes); d=((x[:,None,:]-x[None,:n,:])**2).sum(2); near=np.exp(-d/.20)
    return p,h,rot,modes,z,b,near

def _emit23(cache, yw,tw,rw):
    p,h,rot,m,z,b,near=cache; n=len(m); M=((1-tw)*p[:,1:]+tw*h[:,1:])@rot.T
    if rw:
        # Rescale for the frozen kernel's dimension-normalized dot product.
        q=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]
        design=np.c_[q*np.sqrt((q.shape[1]+near.shape[1])/q.shape[1]),rw*np.sqrt(q.shape[1]+near.shape[1])*near]
    else: design=np.c_[M/np.maximum(m.std(0),1e-5),.035*z,.035*b]
    return {'y_hat':(1-yw)*p[n:,0]+yw*h[n:,0],'trajectory_design':design}
