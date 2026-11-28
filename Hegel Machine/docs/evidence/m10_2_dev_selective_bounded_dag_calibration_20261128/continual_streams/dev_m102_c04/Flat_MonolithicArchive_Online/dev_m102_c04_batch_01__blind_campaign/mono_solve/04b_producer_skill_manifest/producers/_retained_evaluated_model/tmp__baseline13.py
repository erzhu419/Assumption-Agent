import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


def _linear(u):
    return np.column_stack((np.ones(len(u)), u))


def _fit(u, v):
    return np.linalg.lstsq(u, v, rcond=1e-10)[0]


def _pair(u):
    a,b,c,d=u.T
    return np.column_stack((np.ones(len(u)),a*b,a*d,b*c,c*d))


def _cos(u):
    return _linear(np.cos(2.4*u))


def _coordinates(study, x):
    u,v=(study['panel_a'][k] for k in ('u','v'))
    design=_linear(u)
    mask=np.ones(len(u),bool)
    for _ in range(8):
        w=_fit(design[mask],v[mask]); e=np.linalg.norm(design@w-v,axis=1)
        m=e<=max(4*np.median(e),1e-9)
        if m.sum()<8 or np.array_equal(m,mask): break
        mask=m
    offset, scale=np.array([_fit(design[mask][:,[0,i+1]],v[mask,i]) for i in range(4)]).T
    z=(x-offset)/scale
    b=study['panel_b']; w=_fit(_linear(b['u']),b['v'])
    forward=_linear(z)@w; backward=np.linalg.solve(w[1:].T,(z-w[0]).T).T
    c=study['panel_c']; u=c['u']; q=_fit(_pair(u),c['v1']); h=_fit(_cos(u),c['v2']);
    return z,forward,backward,_pair(z)@q,_cos(z)@h,_pair(forward)@q,_cos(forward)@h,_pair(backward)@q,_cos(backward)@h


def _kernels(study, x, n):
    coords=_coordinates(study,x)
    D=[]; G=[]; groups=[]
    def add(K, group):
        G.append(K);groups.append(group)
    add(np.ones((len(x),len(x))),0)
    for j,z in enumerate(coords):
        z=(z-z[:n].mean(0))/np.maximum(z[:n].std(0),1e-8)
        dot=z@z.T/z.shape[1]; d=cdist(z,z,'sqeuclidean')/z.shape[1]
        if j in (0,1,2):
            for degree in (1,2,3,4): add(dot**degree,0 if j==0 else 1)
            for a in (.45,.85,1.5,2.8):
                add(np.exp(-.5*d/a**2),1)
            for a in (.55,1.15):
                per=np.exp(-.5*(z[:,:,None]-z.T[None,:,:])**2/a**2)
                add(per.mean(1),1)
                add(sum(per[:,i,:]*per[:,k,:] for i in range(4) for k in range(i))/6,1)
            add((1+np.sqrt(3*d))*np.exp(-np.sqrt(3*d)),1)
            for i in range(4):
                a=(z[:,i,None]-z[None,:,i])**2
                add(np.exp(-a/2),1)
                z1=np.tanh(1.25*z[:,i]); add(z1[:,None]*z1,1)
        else:
            add(dot,2);add(dot**2,2)
            for a in (.65,1.35):add(np.exp(-.5*d/a**2),2)
            dz=cdist(coords[0],coords[0],'sqeuclidean')/4
            add(np.exp(-.5*(d+dz)/1.2**2),2)
    return np.asarray(G),np.asarray(groups)


def _gp(G, groups, b, settings):
    n=len(b); b=np.asarray(b).reshape(n,-1); mu=b.mean(0); scale=np.maximum(b.std(0),1e-12); bs=(b-mu)/scale
    K=np.ascontiguousarray(G[:,:n,:n]); flat=K.reshape(len(K),-1)
    eye=np.eye(n); pred=[]
    for bias in settings:
        prior=np.array([0.18,0.035,0.012])[groups].copy(); prior[groups==2]*=bias
        center=np.log(np.r_[prior,.03]); shrink=np.r_[np.full(len(prior),.31),.24]
        def objective(p, prediction=False):
            w=np.exp(p); C=(w[:-1]@flat).reshape(n,n)+w[-1]*eye
            cf=cho_factor(C,lower=True,check_finite=False); aa=cho_solve(cf,bs,check_finite=False)
            if prediction:return ((w[:-1]@G[:,n:,:n].reshape(len(K),-1)).reshape(-1,n)@aa)*scale+mu
            delta=p-center; inverse=cho_solve(cf,eye,check_finite=False)
            g=inverse-aa@aa.T/bs.shape[1]
            val=np.log(np.diag(cf[0])).sum()+.5*np.sum(bs*aa)/bs.shape[1]+.5*np.dot(shrink,delta**2)
            grad=.5*w*np.r_[flat@g.ravel(),np.trace(g)]+shrink*delta
            return val,grad
        opt=minimize(objective,center,jac=True,method='L-BFGS-B',bounds=[(-15,5)]*len(center),options={'maxiter':140,'ftol':2e-8})
        pred.append(objective(opt.x,True))
    return np.mean(pred,axis=0)


def solve_batch(study):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):
        d=study['train']; n=len(d['x']); x=np.r_[d['x'],study['test']['x']]
        G,groups=_kernels(study,x,n)
        y=_gp(G,groups,d['y'],(.2,1.5))[:,0]
        _,sigma,vt=np.linalg.svd(d['traj'],full_matrices=False); V=vt[:3]
        a=d['traj']@V.T
        state=np.column_stack([_gp(G,groups,a[:,j],(.2,1.5))[:,0] for j in range(3)])@V
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
