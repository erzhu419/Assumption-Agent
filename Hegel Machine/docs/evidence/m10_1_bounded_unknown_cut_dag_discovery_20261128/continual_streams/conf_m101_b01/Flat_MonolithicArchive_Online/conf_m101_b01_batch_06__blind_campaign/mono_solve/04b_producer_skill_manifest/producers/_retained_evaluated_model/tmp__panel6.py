import numpy as np,pickle
from scipy.optimize import minimize_scalar
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; ij=np.triu_indices(4,1)
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def af(x):return np.c_[np.ones(len(x)),x]
def cross(x):return x[:,ij[0]]*x[:,ij[1]]
def score(d):
 e=abs(d[52:]@ls(d[:52],Y[:52])-Y[52:]);return np.round([e[:,0].mean(),e[:,1:].mean()],5)
def fitfun(p,k,fn,lo,hi):
 u,v=p['u'],p[k]
 def loss(z):
  d=fn(u,z); q=d@np.linalg.pinv(d); return np.mean(((q@v-v)/(1-q.diagonal()[:,None]))**2)
 opt=minimize_scalar(loss,bounds=(lo,hi),method='bounded'); z=opt.x; w=ls(fn(u,z),v); return z,opt.fun,w
if __name__=='__main__':
 for p,k,F in [(b,'v',erf),(b,'v',np.tanh),(b,'v',np.sin),(b,'v',np.arctan),(c,'v2',np.cos)]:
  for typ in ['aff+n','n']:
   fn=lambda u,z:np.c_[af(u) if typ=='aff+n' else np.ones(len(u)),F(z*u)]
   z,e,w=fitfun(p,k,fn,.2,3.5);print(k,F.__name__,typ,z,e,w.round(3).tolist())
 u,v=a['u'],a['v']; d=af(u); keep=np.arange(len(u))
 for _ in range(8):
  w=ls(d[keep],v[keep]); e=np.linalg.norm(d@w-v,axis=1); keep=np.argsort(e)[:45]
 print('A trimmed affine coeff',w.round(4)); print('A residuals',np.sort(e).round(4)); E=v-d@w
 for F in [lambda x:x*x,lambda x:x**3,np.sin,np.cos,np.tanh,erf]:
  for xs in [u,np.roll(u,1,axis=1),u+np.roll(u,1,axis=1),u-np.roll(u,1,axis=1)]:
   D=np.c_[d,F(xs)]; W=ls(D[keep],v[keep]); print(getattr(F,'__name__',''),np.mean((D[keep]@W-v[keep])**2).round(6),np.diag(W[5:]).round(3),end='; ')
  print()
 print('A quadratic',ls(np.c_[d,u*u,cross(u)][keep],v[keep]).round(3))
