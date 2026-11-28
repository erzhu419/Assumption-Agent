import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit

def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def D(x):return np.c_[np.ones(len(x)),x]
def calibrate(panel,x):
 u,v=panel['u'],panel['v']; rng=np.random.default_rng(45); w=np.roll(u,2,axis=1)
 def gram(a,b,k):return D(np.c_[a,np.tanh(k[0]*a),np.sin(k[1]*b)])
 def coeff(k,g):return np.array([ls(gram(u[g,j],w[g,j],k),v[g,j]) for j in range(4)])
 def fore(z,k,c):return c[:,0]+z*c[:,1]+np.tanh(k[0]*z)*c[:,2]+np.sin(k[1]*np.roll(z,2,axis=1))*c[:,3]
 k=np.array([1.8,2.4]); g=np.arange(len(u)); a=[gram(u[:,j],w[:,j],k) for j in range(4)]
 c=min((coeff(k,i) for i in (rng.choice(len(u),9,False) for _ in range(100))),key=lambda c:np.median(np.max(abs(fore(u,k,c)-v),axis=1)))
 err=np.max(abs(fore(u,k,c)-v),axis=1); g=err<max(.001,6*np.median(err)); k=least_squares(lambda k:(fore(u[g],k,coeff(k,g))-v[g]).ravel(),k,bounds=([1.4,2],[2.2,2.8])).x; c=coeff(k,g)
 z=(x-c[:,0])/c[:,1]; j=np.arange(4)
 for i in range(12):
  J=np.zeros((len(z),4,4)); J[:,j,j]=c[:,1]+k[0]*c[:,2]*(1-np.tanh(k[0]*z)**2);J[:,j,(j+2)%4]=k[1]*c[:,3]*np.cos(k[1]*np.roll(z,2,axis=1));z-=np.linalg.solve(J,(fore(z,k,c)-x)[...,None])[...,0]
 return z

def _branches(panel, z):
    u=panel['u']
    def f1(a,k):
        return D(np.c_[np.tanh(k[0]*a), a+k[1]*a/(1+.22*a*a)])
    def f2(a,k):
        return D(np.c_[a,np.arctan(k[0]*a)+k[1]*a*np.roll(a,1,axis=1)])
    out=[]
    for key, design, init, bounds in [('v1',f1,[1.75,-1.4],([1.4,-2],[2.2,-.8])),('v2',f2,[1.08,.27],([.8,.1],[1.4,.5]))]:
        v=panel[key]
        def residual(k):
            a=design(u,k)
            return (a@ls(a,v)-v).ravel()
        k=least_squares(residual,init,bounds=bounds,ftol=1e-11).x
        out.append(design(z,k)@ls(design(u,k),v))
    return out

def _readouts(p,q):
    cross=(p[:,:,None]*q[:,None,:]).reshape(len(p),16)
    dot=np.sum(p*q,axis=1)
    return D(np.c_[p,q,dot]), D(np.c_[p,q,dot]), D(np.c_[q,cross])

def _forecast(t,obs,fast,slow):
    n=len(obs)
    def matrix(k,i):
        h=-np.expm1(-t[:,None]*k)
        return np.concatenate([fast[i,None,:]*h[None,:,:1],slow[i,None,:]*h[None,:,1:]],axis=2)
    def residual(k):
        a=matrix(k,slice(n)).reshape(obs.size,-1)
        return a@ls(a,obs.ravel())-obs.ravel()
    fit=least_squares(residual,[1.83,.42],bounds=([1.1,.1],[3,.9]),ftol=1e-11)
    a=matrix(fit.x,slice(n)).reshape(obs.size,-1)
    return matrix(fit.x,slice(n,None))@ls(a,obs.ravel())

def solve_batch(study: dict) -> dict:
    tr=study['train']; n=len(tr['x'])
    z=calibrate(study['panel_a'],np.r_[tr['x'],study['test']['x']])
    p,q=_branches(study['panel_c'],z)
    scalar,fast,slow=_readouts(p,q)
    y=scalar[n:]@ls(scalar[:n],tr['y'])
    state=_forecast(np.asarray(study['eval_times']),tr['traj'],fast,slow)
    last=state[:,-1]
    return dict(y_hat=y,state_pred=state,diagnostics_vec=np.array([y.mean(),y.std(),last.mean(),last.std()]))
