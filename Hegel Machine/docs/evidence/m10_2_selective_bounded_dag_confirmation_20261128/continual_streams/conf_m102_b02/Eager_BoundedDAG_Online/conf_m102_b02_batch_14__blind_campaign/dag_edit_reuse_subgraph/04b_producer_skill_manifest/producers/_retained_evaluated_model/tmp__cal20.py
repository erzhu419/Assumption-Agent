import numpy as np
from scipy.optimize import least_squares, minimize_scalar

def fit(f,y): return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def D(x): return np.c_[np.ones(len(x)),x]
def af(x): return D(np.c_[x,x*np.roll(x,-1,axis=1),x**3])
def coords(s,x,verbose=False):
 p=s['panel_a']; u=p['u'];v=p['v']; k=np.linalg.norm(u-v,axis=1)<3
 A=af(u); c=fit(A[k],v[k]); z=np.linalg.solve(c[1:5].T,(x-c[0]).T).T
 for _ in range(20):
  J=c[1:5]+np.roll(z,-1,axis=1)[:,:,None]*c[5:9]+np.roll(z[:,:,None]*c[5:9],1,axis=1)+3*z[:,:,None]**2*c[9:13]
  delta=np.linalg.solve(J.transpose(0,2,1),(af(z)@c-x)[...,None])[...,0]; z-=np.clip(delta,-.3,.3)
 if verbose:print('A max residual',np.max(abs(A[k]@c-v[k])),np.max(abs(af(z)@c-x)), 'coef', c.round(4))
 p=s['panel_b']; u=p['u'];v=p['v'];
 candidates=[]
 for kind in range(7):
  def fun(x,q):
   cos=np.cos(q[0]*x); sin=np.sin(q[0]*x)
   return D([cos,cos+x*q[1],np.cos(q[0]*x+q[1]),np.c_[cos,sin],np.c_[cos,x],np.c_[cos,x*np.roll(x,-1,1)],np.c_[cos,np.cos(q[0]*x)*np.roll(x,-1,1)]][kind])
  def err(q):
   f=fun(u,q); return (f@fit(f,v)-v).ravel()
  res=min([least_squares(err,[k,0.15],bounds=([.05,-3],[9,3])) for k in [1,2,3,4,5,7,8]],key=lambda r:r.cost)
  q=res.x; f=fun(u,q); cb=fit(f,v); candidates.append((res.cost,fun(z,q)@cb))
  if verbose:print('B',kind,q,np.sqrt(2*res.cost/v.size), 'max',abs(res.fun).max(), 'coef', cb.round(3) if res.cost<1e-4 else '')
 return z,min(candidates,key=lambda v:v[0])[1]
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b=coords(s,np.r_[s['train']['x'],s['test']['x']],True); print('ranges',z.min(0),z.max(0),b.min(0),b.max(0))
