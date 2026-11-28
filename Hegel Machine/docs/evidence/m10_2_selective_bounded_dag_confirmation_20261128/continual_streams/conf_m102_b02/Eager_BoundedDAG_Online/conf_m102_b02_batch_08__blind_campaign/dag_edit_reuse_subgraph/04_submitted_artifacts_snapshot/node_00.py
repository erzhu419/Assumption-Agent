import numpy as np
from scipy.optimize import least_squares

def D(x): return np.c_[np.ones(len(x)), x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def A(x): return D(np.c_[x,x*np.roll(x,-1,axis=1),x**3])
def coords(s,x):
    p=s['panel_a']; u,v=p['u'],p['v']; F=A(u)
    rng=np.random.default_rng(42)
    def err(w): return np.linalg.norm(F@w-v,axis=1)
    w=fit(F,v)
    if np.median(err(w))>.001:
        w=min((fit(F[i],v[i]) for i in [rng.choice(len(F),17,replace=False) for j in range(80)]),key=lambda w: np.median(err(w)))
    r=err(w); keep=r<max(.001,4*np.median(r)); w=fit(F[keep],v[keep])
    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
    for j in range(20):
        J=w[1:5]+np.roll(z,-1,axis=1)[:,:,None]*w[5:9]+np.roll(z[:,:,None]*w[5:9],1,axis=1)+3*z[:,:,None]**2*w[9:13]
        dz=np.linalg.solve(J.transpose(0,2,1),(A(z)@w-x)[...,None])[...,0]; z-=np.clip(dz,-.45,.45)
        if np.max(np.abs(dz))<1e-10:break
    return z

def parts(s):
    z=coords(s,np.r_[s['train']['x'],s['test']['x']]); p=s['panel_b']; u,v=p['u'],p['v']
    def H(x,q): return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1])
    def e(q):
        f=H(u,q); return (f@fit(f,v)-v).ravel()
    q=least_squares(e,[.35,1.5],bounds=([0,.5],[.9,3])).x
    p=s['panel_c'];  P=lambda x:D(x*np.roll(x,-1,axis=1))
    return z,H(z,q)@fit(H(u,q),v),P(z)@fit(P(p['u']),p['v1'])

def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; z,b,c=parts(s); p=s['panel_c']; u,v=p['u'],p['v2']
    from scipy.optimize import minimize_scalar
    def H(x,k): return D(np.c_[x,np.cos(k*x)])
    def loss(k):
        f=H(u,k); return np.mean((f@fit(f,v)-v)**2)
    k=minimize_scalar(loss,bounds=(.2,4),method='bounded').x
    d=H(z,k)@fit(H(u,k),v)
    t=D(np.c_[z,b*np.roll(b,-1,axis=1),d[:,0]*d[:,1]*d[:,2],b*np.roll(c,1,axis=1),(.3*z+b)**3]); n=len(s['train']['y'])
    return {'y_hat':t[n:]@fit(t[:n],s['train']['y']), 'trajectory_design':t}
