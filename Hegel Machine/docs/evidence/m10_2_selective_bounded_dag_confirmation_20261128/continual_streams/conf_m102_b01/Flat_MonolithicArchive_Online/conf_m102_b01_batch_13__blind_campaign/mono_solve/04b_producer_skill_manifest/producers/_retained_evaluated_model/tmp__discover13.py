import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
ls=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
aff=lambda x: np.c_[np.ones(len(x)),x]
p=s['panel_a']; u,v=p['u'],p['v']; one=np.ones(len(u)); rng=np.random.default_rng(1)
for j in range(4):
 z=u[:,j]; q=u[:,(j+1)%4]; r=u[:,(j-1)%4]; h=u[:,(j+2)%4]
 for name,d in [('diag',[one,z]),('aff',[one,*u.T]),('next',[one,z,q,z*q,z**3]),('prev',[one,z,r,z*r,z**3]),('sat',[one,z,np.tanh(1.8*z),np.sin(2.4*h)]),('full',[one,*u.T,*[u[:,i]*u[:,k] for i,k in combinations_with_replacement(range(4),2)],*u.T**3])]:
  d=np.array(d).T; best=None
  for rep in range(30):
   ix=rng.choice(len(u),max(d.shape[1]+2,28),replace=False)
   for _ in range(8): ix=np.argsort(abs(d@ls(d[ix],v[ix,j])-v[:,j]))[:40]
   w=ls(d[ix],v[ix,j]); e=abs(d@w-v[:,j]); score=np.mean(np.sort(e)[:40]**2)
   if best is None or score<best[0]: best=(score,e,w)
  _,e,w=best; print('A',j,name,np.quantile(e,[.5,.8,.9,1]),w if name!='full' else '')
for key in ('panel_b','panel_c'):
 p=s[key]; u=p['u']; print(key,np.column_stack(list(p.values()))[:8]);
 for k,v in p.items():
  if k=='u':continue
  print(k,'norm',np.quantile(np.linalg.norm(v,axis=1),[0,.25,.5,.75,1]),'sv',np.linalg.svd(v-v.mean(0),compute_uv=False))
  for name,d in [('aff',u),('dead',np.sign(u)*np.maximum(abs(u)-.35,0)**1.5),('tanh',np.tanh(1.75*u)+.30*np.sin(2.10*np.roll(u,1,axis=1))),('cross',np.c_[u[:,0]*u[:,1],u[:,0]*u[:,3],u[:,1]*u[:,2],u[:,2]*u[:,3]]),('quad',np.c_[u,*[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]]),('square',u*u),('sin',np.sin(2.15*u)),('cos',np.cos(2.15*u)),('asinh',np.arcsinh(2.15*u)),('rat',u/(1+.22*u*u))]:
   d=aff(d); cf=ls(d,v); ri=np.sqrt(np.mean((d@cf-v)**2)); H=np.sum(np.linalg.pinv(d).T*d,1); loo=np.sqrt(np.mean(((d@cf-v)/(1-H[:,None]))**2)); print(name,round(ri,5),round(loo,5))
t=s['eval_times']; T=s['train']['traj'];
for mode in [0,1]:
 def basis(r):
  B=-np.expm1(-t[:,None]*r); return np.c_[np.ones(len(t)),B] if mode else B
 def err(r):
  B=basis(r);return (B@ls(B,T.T)-T.T).ravel()
 o=least_squares(err,[.4,2],bounds=(.002,10)); B=basis(o.x); W=ls(B,T.T).T; print('rates',o.x,'rmse',np.std(o.fun),'amplitudes',W[:6]);
 if not mode: np.save('/tmp/amps.npy',W)
