import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
u,v=[s['panel_a'][k] for k in ('u','v')]; q=np.roll(u,-1,axis=1); r=np.roll(u,1,axis=1)
for arr in [[u],[u,q,u*q,u**3],[u,q,u*q,u**3,r,u*r,u*u,q*q],[u,q,u*q,u**3,r,u*r,u*u,q*q,np.roll(u,2,axis=1)]]:
 D=np.stack([np.ones_like(u)]+arr,axis=-1); ix=np.ones(len(u),bool)
 for i in range(12):
  co=np.array([fit(D[ix,j],v[ix,j]) for j in range(4)]); e=np.linalg.norm(np.einsum('njk,jk->nj',D,co)-v,axis=1); ix=e<max(.002,3*np.median(e))
 print('A',ix.sum(),np.quantile(e,[0,.5,.65,.8,1]),co)
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u']; v=s[p][k]; q=np.roll(u,-1,axis=1)
 funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'gauss':lambda x:np.exp(-x*x),'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:1/(1+x*x),'linear':lambda x:x}
 print(p,k)
 for name,f in funcs.items():
  for typ in (0,1,2):
   def obj(w):
    a=aff(f(w*u)); a=np.c_[a,u*q] if typ==1 else (np.c_[a,u] if typ==2 and name!='linear' else a);return np.sqrt(np.mean((a@fit(a,v)-v)**2))
   rr=minimize_scalar(obj,bounds=(.1,4),method='bounded');print(name,typ,round(rr.x,4),round(rr.fun,5),end='; ')
  print()
 print('linear coeff',fit(aff(u),v))
t=s['train']['traj']; print('SVD',np.linalg.svd(t,compute_uv=False));print('summary',s['train']['summary']);print('y affine traj',np.sqrt(np.mean((aff(t)@fit(aff(t),s['train']['y'])-s['train']['y'])**2)))
