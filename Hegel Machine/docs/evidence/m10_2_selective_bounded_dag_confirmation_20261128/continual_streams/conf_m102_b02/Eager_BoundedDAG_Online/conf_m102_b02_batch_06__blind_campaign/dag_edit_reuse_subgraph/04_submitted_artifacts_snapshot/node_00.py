"""Predict scalar and trajectory outputs from calibrated measurements."""
import numpy as np
from scipy.optimize import least_squares
def D(x):return np.c_[np.ones(len(x)),x]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def cal(s,x):
 p=s['panel_a']; F=D(p['u']); y=p['v']; keep=np.ones(len(F),bool)
 for _ in range(8):
  c=fit(F[keep],y[keep]); e=np.linalg.norm(F@c-y,axis=1);keep=e<max(.001,3*np.median(e))
 return np.linalg.solve(c[1:].T,(x-c[0]).T).T,c,e
def AF(u): return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
def coords(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; F=AF(u); _,_,e=cal(s,x); w=fit(F[e<2],v[e<2]); z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for j in range(12):
  J=np.broadcast_to(w[1:5],(len(z),4,4)).copy(); K=w[5:9]; rr=np.roll(z,-1,axis=1)
  J+=rr[:,:,None]*K[None,:,:]+np.roll(z[:,:,None]*K[None,:,:],1,axis=1)+3*z[:,:,None]**2*w[None,9:13,:]
  dz=np.linalg.solve(J.transpose(0,2,1),(AF(z)@w-x)[...,None])[...,0];z-=np.clip(dz,-.4,.4)
 return z

def parts(study):
    tr=study['train']; z=coords(study,np.r_[tr['x'],study['test']['x']]); p=study['panel_b']; u=p['u'];v=p['v']
    def H(x,q):return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1])
    def e(q):
        f=H(u,q);return (f@fit(f,v)-v).ravel()
    q=least_squares(e,[.35,1.5],bounds=([.15,.6],[.6,2.5])).x
    b=H(z,q)@fit(H(u,q),v)
    return z,b


def features(study):
    z,b=parts(study)
    p=study['panel_c'];u=p['u'];v=p['v2']
    def H(x,q):
        return D(x+q[0]*(np.logaddexp(q[1]*x,-q[1]*x)-np.log(2.)))
    def err(q):
        f=H(u,q);return (f@fit(f,v)-v).ravel()
    q=least_squares(err,[1.,1.],bounds=([.01,.01],[20.,5.])).x
    c=H(z,q)@fit(H(u,q),v)
    cz=D(z*np.roll(z,-1,axis=1))@fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
    return z,b,c,cz

def kernel(x):
    dot=x@x.T/x.shape[1]
    dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0.)
    return (1+dot)**3+np.exp(-.3*dist)

def run_node(inputs:dict)->dict:
    s=inputs['__input__']; z,b,c,p=features(s); y=s['train']['y']; n=len(y)
    f=np.arcsinh(2*np.c_[z,b,c]); k=kernel(f)
    yhat=k[n:,:n]@np.linalg.solve(k[:n,:n]+.002*np.eye(n),y)
    return {'y_hat':yhat,'z':z,'b':b,'c':c,'p':p}
