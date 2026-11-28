from probe import *
from itertools import product
u=s['panel_a']['u']; v=s['panel_a']['v']; d=4; ii,jj=np.triu_indices(d)
def robust(h,v):
 k=np.ones(len(v),bool)
 for _ in range(7):
  if h.ndim==2: c=fit(h[k],v[k]); pred=h@c
  else: c=fit(h[k].reshape(-1,h.shape[-1]),v[k].ravel()); pred=h@c
  e=np.linalg.norm(pred-v,axis=1); new=e<=max(4*np.median(e),0.001)
  if np.array_equal(new,k):break
  k=new
 return np.sqrt(np.mean(e[k]**2)/4),c,k
np.set_printoptions(precision=4,suppress=True)
print('A first',np.c_[u[:12],v[:12]]); print('A',np.sort(np.linalg.norm(v,axis=1))[-8:]);
for flip in [0,1]:
 x,y=(u,v) if not flip else (v,u)
 for pairs in [x[:,ii]*x[:,jj],x*x,x*abs(x),np.c_[x*x,x**3],np.c_[x*x,x**3,x**4],np.empty((len(x),0))]:
  h=aff(np.c_[x,pairs]); r,c,k=robust(h,y);print('A full',flip,h.shape[-1],r,k.sum());
  if h.shape[1]==5: print(c)
 for shift in [-1,1,2]:
  for fun in [lambda u:u**3,lambda u:u*abs(u),lambda u:u**2,lambda u:np.sin(2.15*u)]:
   r=np.roll(x,shift,axis=1); base=np.stack([x,r,x*r,fun(x)],-1); cols=np.concatenate([np.eye(d)[None,:,:,None]*np.c_[np.ones((len(x),d,1)),] if False else np.empty((len(x),d,0)),base],-1)
   hh=np.concatenate([np.broadcast_to(np.eye(d),(len(x),d,d)),(np.eye(d)[None,:,:,None]*cols[:,:,None,:]).reshape(len(x),d,-1)],-1); e,c,k=robust(hh,y); print('A shift',flip,shift,e)
