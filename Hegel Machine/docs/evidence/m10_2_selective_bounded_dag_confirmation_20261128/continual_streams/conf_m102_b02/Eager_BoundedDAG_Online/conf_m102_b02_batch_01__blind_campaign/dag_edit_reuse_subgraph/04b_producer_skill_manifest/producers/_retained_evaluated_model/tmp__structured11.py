from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


def _profile(x, study):
    """Estimate an additive/interacting coordinate map from training rows only."""
    t = study['train']; n = len(t['x']); a = np.asarray(t['traj'], float)
    modes = np.linalg.svd(a, full_matrices=False)[2][:2]
    y = np.c_[t['y'], a @ modes.T]
    center = y.mean(0); scale = np.maximum(y.std(0), 1e-8)
    y = (y - center) / scale
    blocks = []
    def add(d, interaction=True):
        d = d - d[:n].mean(0)
        d = d / np.maximum(d[:n].std(0), 1e-7)
        g = d @ d[:n].T / d.shape[1]
        blocks.append(g)
        if interaction:
            blocks.append(g*g)
    for w in (x,np.arcsinh(3.5*x),np.tanh(2*x),x*x,x**3/(1+0.8*x*x),np.sin(3*x),np.cos(2.5*x)):
        add(w)
    for pk,vk,func in (
        ('panel_b','v',lambda u:u**3/(1+0.8*u*u)),
        ('panel_c','v1',lambda u:np.arcsinh(3.5*u)),
        ('panel_c','v2',lambda u:u*u/(1+0.165*u*u))):
        p=study[pk]
        def des(u): return np.c_[np.ones(len(u)),u,func(u)]
        c=np.linalg.lstsq(des(p['u']),p[vk],rcond=None)[0]
        add(des(x)@c)
    i,j=np.triu_indices(x.shape[1],1)
    for w in (x,np.arcsinh(3.5*x),x**3/(1+0.8*x*x)):
        for z in (w[:,i]+w[:,j],w[:,i]-w[:,j]):
            add(np.arcsinh(2*z)); add(z*z,False)
    for k in (0.25,0.8):
        dist=np.sum((x[:,None,:]-x[None,:n,:])**2,axis=2)
        blocks.append(np.exp(-k*dist))
    blocks.append(np.ones((len(x),n)))
    G=np.ascontiguousarray(blocks); A=np.ascontiguousarray(G[:,:n,:].reshape(len(G),-1)); eye=np.eye(n)
    def fit(target):
        def objective(theta):
            w=np.exp(theta); k=(w[:-1]@A).reshape(n,n)+w[-1]*eye
            c=cho_factor(k,check_finite=False); inv=cho_solve(c,eye,check_finite=False)
            alpha=inv@target; dg=inv-alpha@alpha.T/target.shape[1]
            delta=theta+4
            loss=(np.sum(target*alpha)/target.shape[1]+2*np.log(c[0].diagonal()).sum())/n+0.00025*(delta@delta)
            grad=np.r_[A@dg.ravel(),np.trace(dg)]*w/n+0.0005*delta
            return loss,grad
        o=minimize(objective,np.full(len(G)+1,-4.),jac=True,bounds=[(-14,4)]*(len(G)+1),method='L-BFGS-B',options={'maxiter':180})
        w=np.exp(o.x); k=np.einsum('a,aij->ij',w[:-1],G); c=cho_factor(k[:n]+w[-1]*eye,check_finite=False)
        return k@cho_solve(c,target,check_finite=False)
    common=fit(y)
    separate=np.column_stack([fit(y[:,k:k+1])[:,0] for k in range(y.shape[1])])
    return .5*(common+separate)


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; x=_latent(study); n=len(study['train']['x'])
    p=_profile(x,study)
    z=np.c_[.15*p,.065*x]
    return {'z':z[:n].copy(),'zt':z[n:].copy()}
