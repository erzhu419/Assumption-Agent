import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _cycle(x):
    return x * np.roll(x, -1, axis=1)


def _panels(u, v1, v2, x):
    def calibrate(f, v):
        return f(x) @ np.linalg.lstsq(f(u), v, rcond=None)[0]
    return calibrate(lambda a: _one(_cycle(a)), v1), calibrate(lambda a: _one(np.c_[a, a*a]), v2)


def _kernels(x, z, b, c, d, n):
    groups = []
    def std(a):
        return (a-a[:n].mean(0))/np.maximum(a[:n].std(0), 1e-10)
    def add(a):
        a = std(a)
        groups.append(a @ a.T / a.shape[1])
    panels = [x, z, b, c, d]
    for f in panels:
        for j in range(f.shape[1]):
            q = f[:, j:j+1]
            a = std(q)
            add(a)
            add(a*a)
            add(np.c_[np.sin(a), np.tanh(1.6*a), a/(1+2*a*a), a**3])
            add(np.c_[np.cos(2*a), np.sin(2.5*a), a**4])
    for p, f in enumerate(panels):
        for r, g in enumerate(panels[:p+1]):
            f0, g0 = std(f), std(g)
            ix = [(i,j) for i in range(4) for j in range(4) if r!=p or i<j]
            for subset in (ix, [(i,j) for i,j in ix if (i-j)%4 in (1,3)]):
                add(np.column_stack([f0[:,i]*g0[:,j] for i,j in subset]))
            if p < 4 and r < 3:
                add(np.column_stack([np.tanh(1.4*f0[:,i])*g0[:,j] for i,j in ix]))
    for p, f in enumerate(panels[:4]):
        a=std(f); k=a@a.T/4
        groups.extend([(1+k)**3/8, (1+k)**4/16])
        for g in panels[:p]:
            h=std(g); l=h@h.T/4
            groups.append((1+k+l)**3/12)
    a=std(np.c_[x,z,b,c]); k=a@a.T/a.shape[1]; dist=k.diagonal()[:,None]+k.diagonal()[None,:]-2*k
    groups.extend([(1+k)**2/4, (1+k)**3/8, np.exp(-.3*np.maximum(dist,0)), np.exp(-np.maximum(dist,0))])
    return np.array(groups)


def _predict(k, y, noise, penalty):
    n=len(y); m=len(k); mu=y.mean(0); scale=np.maximum(y.std(0),1e-10); y=(y-mu)/scale
    a=np.concatenate((np.eye(n)[None],k[:,:n,:n])); flat=np.ascontiguousarray(a.reshape(m+1,-1)); cross=k[:,n:,:n]; inv0=np.eye(n)
    prior=np.log(np.r_[noise,np.full(m,1./m)])
    def calc(theta):
        v=np.exp(theta); cov=(np.einsum('i,ij->j',v,flat)).reshape(n,n)+1.
        cf=cho_factor(cov,check_finite=False)
        alpha=cho_solve(cf,y,check_finite=False)
        ic=cho_solve(cf,inv0,check_finite=False)
        loss=np.log(cf[0].diagonal()).sum()+.5*np.sum(alpha*y)/y.shape[1]
        grad=.5*v*(np.einsum('ij,j->i',flat,(ic-alpha@alpha.T/y.shape[1]).ravel()))
        delta=theta-prior
        return loss+penalty*(delta@delta),grad+2*penalty*delta
    theta=minimize(calc,prior,jac=True,method='L-BFGS-B',bounds=[(-12.,1.)]+[(-15.,3.)]*m,options={'maxiter':85,'ftol':2e-7}).x
    v=np.exp(theta); cov=(np.einsum('i,ij->j',v,flat)).reshape(n,n)+1.
    alpha=cho_solve(cho_factor(cov,check_finite=False),y,check_finite=False)
    return ((1.+np.einsum('k,kij->ij',v[1:],cross))@alpha)*scale+mu


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; z=inputs['node_00']['z']; b=inputs['node_00']['b']
    train=s['train']; x=np.vstack((train['x'],s['test']['x'])); n=len(train['y'])
    panel=s['panel_c']; c,d=_panels(panel['u'],panel['v1'],panel['v2'],b)
    k=_kernels(x,z,b,c,d,n)
    times=s['eval_times']
    h=np.column_stack((-np.expm1(-1.56*times),-np.expm1(-.38*times),np.ones(len(times))))
    q=np.linalg.qr(h)[0]
    modes=train['traj']@q
    # Share covariance relevance between the scalar and dominant time mode.
    joint=np.column_stack((train['y'],modes[:,0]))
    prediction=0.
    for penalty, weight in ((0.004, 0.5), (0.002, 0.5)):
        prediction=prediction+weight*np.column_stack((
            _predict(k,joint,.004,penalty),
            _predict(k,modes[:,1:2],.004,penalty),
            _predict(k,modes[:,2:3],.04,penalty)))
    y_hat=prediction[:,0]
    state_pred=prediction[:,1:]@q.T
    final=state_pred[:,-1]
    return {'y_hat':y_hat,'state_pred':state_pred,'diagnostics_vec':np.array([y_hat.mean(),y_hat.std(),final.mean(),final.std()])}
