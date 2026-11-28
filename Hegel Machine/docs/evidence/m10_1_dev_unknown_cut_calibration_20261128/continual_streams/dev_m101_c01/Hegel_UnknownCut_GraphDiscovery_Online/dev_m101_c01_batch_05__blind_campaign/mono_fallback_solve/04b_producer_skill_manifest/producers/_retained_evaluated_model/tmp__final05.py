import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, least_squares
from scipy.spatial.distance import cdist

def affine(panel, key, robust=False):
    u, v = panel['u'], panel[key]
    f = np.column_stack((np.ones(len(u)),u)); ids=np.arange(len(u))
    for _ in range(7 if robust else 1):
        c=np.linalg.lstsq(f[ids],v[ids],rcond=None)[0]
        e=np.linalg.norm(f@c-v,axis=1)
        if robust: ids=np.flatnonzero(e<max(.001,4*np.median(e)))
    return c

def apply(c,q): return q@c[1:]+c[0]

def temporal(t,z):
    def basis(p):
        k,d,w=p
        return np.column_stack((-np.expm1(-k*t),np.exp(-d*t)*np.sin(w*t)))
    def res(p):
        f=basis(p);return (f@np.linalg.lstsq(f,z.T,rcond=None)[0]-z.T).ravel()
    return np.linalg.qr(basis(least_squares(res,[.5,.2,1.1],bounds=([.01,0,.4],[4,2,2])).x))[0]

def blocks(s,x,n,mode):
    a=affine(s['panel_a'],'v',True); b=affine(s['panel_b'],'v'); c=affine(s['panel_c'],'v1')
    q=apply(a,x); p=apply(b,q); v=apply(c,q)
    def scale(u): return (u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-6)
    def gram(u): return u@u.T
    ans=[np.ones((len(x),len(x)))]; names=['const']
    def add(k,name):
        ans.append(k);names.append(name)
    for key,u in [('A',q),('raw',x)]:
        g=gram(scale(u))/4
        for j in range(1,5 if mode!='smooth' else 3): add(g**j,key+str(j))
        d=cdist(scale(u),scale(u),'sqeuclidean')
        for r in (.8,1.6,3.2,6.4): add(np.exp(-d/(2*r*r)),key+'rbf'+str(r))
        d=np.sqrt(5*d)/2
        add((1+d+d*d/3)*np.exp(-d),key+'matern')
    if mode in ('panel','sparse'):
        for key,u in [('A',q),('BA',p),('C',v),('raw',x)]:
            g=gram(scale(u))/4
            for k in (.6,1.3):
                for fn in (np.sin,np.tanh,np.cos):
                    f=fn(k*u); f-=f[:n].mean(0); h=gram(f)
                    add(h,key+fn.__name__+str(k)); add(h*g,key+fn.__name__+'cross'+str(k));add(h*h,key+fn.__name__+'sq'+str(k))
    if mode=='sparse':
        for key,u in [('A',q),('BA',p),('C',v)]:
            u=scale(u)
            for j,f in enumerate(u.T):
                for k in (3,4): add(np.outer(f**k,f**k),key+str(j)+'p'+str(k))
    return np.array(ans),names

def predict(ks,y,prior=.02):
    n=len(y); k=ks[:,:n,:n]; l=ks[:,n:,:n]; h=np.maximum(np.diagonal(k,axis1=1,axis2=2).mean(1),1e-12)
    I=np.eye(n); k=np.concatenate((k/h[:,None,None],I[None])); l=np.concatenate((l/h[:,None,None],np.zeros_like(l[:1]))); mu=y.mean(0); sd=np.maximum(y.std(0),1e-10); z=(y-mu)/sd
    if z.ndim==1:z=z[:,None]
    m=z.shape[1]; kf=k.reshape(len(k),-1); lf=l.reshape(len(l),-1)
    p0=np.full(len(k),-5.);p0[-1]=-6
    def fun(p):
        w=np.exp(p); C=cho_factor((np.einsum('i,ij->j',w,kf)).reshape(n,n),check_finite=False)
        a=cho_solve(C,z,check_finite=False); inv=cho_solve(C,I,check_finite=False)
        return .5*np.sum(a*z)+m*np.log(np.diag(C[0])).sum()+.5*prior*np.sum((p-p0)**2), .5*w*(np.einsum('ij,j->i',kf,(m*inv-a@a.T).ravel()))+prior*(p-p0)
    pp=minimize(fun,p0,jac=True,method='L-BFGS-B',bounds=[(-16,4)]*len(k),options={'maxiter':180,'ftol':1e-9}).x
    w=np.exp(pp); C=cho_factor((np.einsum('i,ij->j',w,kf)).reshape(n,n),check_finite=False); a=cho_solve(C,z,check_finite=False)
    pred=(np.einsum('i,ij->j',w,lf)).reshape(-1,n)@a*sd+mu
    return pred,w


def panel_operator(u, v, quadratic):
    def design(x, w):
        terms = (x, x*x) if quadratic else (x,)
        return np.column_stack((np.ones(len(x)), *terms, np.sin(x*w), np.cos(x*w)))

    def residual(w):
        f = design(u, w)
        return (f @ np.linalg.lstsq(f, v, rcond=None)[0] - v).ravel()

    fit = min((least_squares(residual, np.full(u.shape[1], start),
                             bounds=(.3, 5), max_nfev=120)
               for start in (1., 2., 3.5, 4.5)), key=lambda result: result.cost)
    w = fit.x
    coef = np.linalg.lstsq(design(u, w), v, rcond=None)[0]
    return lambda x: design(x, w) @ coef


def operator_kernels(u, n):
    u = (u - u[:n].mean(0)) / np.maximum(u[:n].std(0), 1e-7)
    g = u @ u.T / u.shape[1]
    return np.array([g, g*g, np.exp(-cdist(u, u, 'sqeuclidean')/8)])


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    n = len(train['x'])
    a = affine(study['panel_a'], 'v', True)
    b = affine(study['panel_b'], 'v')
    q = apply(a, x)
    r = apply(b, q)
    panel = study['panel_c']
    c = panel_operator(panel['u'], panel['v1'], False)
    d = panel_operator(panel['u'], panel['v2'], True)
    extra = np.concatenate([operator_kernels(u, n) for u in (c(q), d(q), c(r), d(r))])
    panel_k = np.concatenate((blocks(study, x, n, 'panel')[0], extra))
    smooth_k = blocks(study, x, n, 'smooth')[0]
    target_panel = predict(panel_k, train['y'])[0][:, 0]
    target_smooth = predict(smooth_k, train['y'])[0][:, 0]
    y_hat = .8*target_panel + .2*target_smooth
    basis = temporal(study['eval_times'], train['traj'])
    state_k = np.concatenate((smooth_k, extra))
    state = predict(state_k, train['traj'] @ basis)[0] @ basis.T
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
