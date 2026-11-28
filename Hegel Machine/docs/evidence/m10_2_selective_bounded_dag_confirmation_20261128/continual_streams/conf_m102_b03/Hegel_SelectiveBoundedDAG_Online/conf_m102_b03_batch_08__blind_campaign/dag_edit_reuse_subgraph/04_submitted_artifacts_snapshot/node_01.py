import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _solve(s, f, report=False):
    y = np.asarray(s['train']['y']); traj = np.asarray(s['train']['traj']); n = len(y)
    z, L, Q, R, p, c = (f[k] for k in ('z','L','Q','R','product','cosine'))
    m = len(z)
    dot = (Q*R).sum(1)
    cross = (p[:,:,None]*c[:,None,:]).reshape(m,-1)
    B = np.c_[np.ones(m),p,c,cross]
    Y = np.c_[y,traj]; beta = np.linalg.lstsq(B[:n],Y,rcond=None)[0]; trend = B@beta
    E = Y-trend[:n]; load = np.linalg.svd(E,full_matrices=False)[2][0]; load /= load[0] if abs(load[0])>.1 else 1
    target = y+(E@load/(load@load)-E[:,0])*.75 if load[0]==1 else y
    mean_blocks = [np.c_[p,c,dot],np.c_[z,p,c,dot],np.c_[p,c,cross],np.c_[z,p,c,cross]]
    Gs = [np.ones((m,m))]+[1+A@A.T/A.shape[1] for A in mean_blocks]
    def dist(A):
        return ((A[:,None,:]-A[None,:,:])**2).sum(2)
    dz,dl,dqr=dist(z),dist(L),dist(np.c_[Q,R]); eye=np.eye(n)
    vals=[]
    def add(K,key,reg):
        for lam in reg:
            fac=cho_factor(K[:n,:n]+lam*eye,check_finite=False)
            a=cho_solve(fac,target,check_finite=False)
            d=cho_solve(fac,eye,check_finite=False).diagonal()
            e=a/d
            score=np.mean(abs(e))+.12*np.sqrt(np.mean(e**2))
            vals.append((score,K[n:,:n]@a,key+(lam,)))
    scales=[.001,.1,1.,10.,100.]; reg=[1e-8,3e-7,1e-5,3e-4,.003]
    for j,D in enumerate([dz,dl,dz+.5*dqr,dqr,dz+.5*dl]):
        for h in [1.15,1.7,2.25,3.,4.,5.5,8.,12.]:
            for k,K in enumerate([np.exp(-D/h**2),(1+D/h**2)**-2]):
                for g,G in enumerate(Gs):
                    for w in scales if g else [1.]:
                        add(K+w*G,('radial',j,h,k,g,w),reg)
    for j,X in enumerate([z,L,np.c_[z,L]]):
        for gamma in [.25,.6,1.2]:
            for degree in [3,4,5,6,8]:
                K=(1+gamma*(X@X.T)/X.shape[1])**degree
                for g,G in enumerate(Gs[1:3],1):
                    for w in [.1,10.]:
                        add(K+w*G,('polynomial',j,gamma,degree,g,w),reg)
    for j,X in enumerate([z,L,np.c_[z,L]]):
        sq=(X[:,None,:]-X[None,:,:])**2
        for h in [.65,1.,1.5,2.4,3.8]:
            H=np.exp(-sq/h**2)
            A=H.mean(2); P=A*A; whole=np.exp(-sq.sum(2)/(2*h)**2)
            for k,K in enumerate([A+.05*whole,A+.15*P+.02*whole,A+.7*P+.08*whole,P+.15*whole]):
                for g,G in enumerate(Gs[1:3],1):
                    for w in [.1,1.,10.]:
                        add(K+w*G,('additive',j,h,k,g,w),reg)
    vals.sort(key=lambda v:v[0]); best=vals[:30]; score=np.array([v[0] for v in best]); weights=np.exp(-12*(score/score[0]-1)); pred=np.average(np.array([v[1] for v in best]),axis=0,weights=weights)
    state=trend[n:,1:]+(pred-trend[n:,0])[:,None]*load[1:]
    if report:
        return pred,state,vals,trend,load
    return pred,state


def run_node(inputs: dict) -> dict:
    s, f = inputs['__input__'], inputs['node_00']
    y, state = _solve(s,f)
    last=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}
