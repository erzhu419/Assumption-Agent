import pickle,numpy as np,itertools
from scipy.special import erf
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def poly(u,d=2):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for k in range(1,d+1) for i in itertools.combinations_with_replacement(range(u.shape[1]),k)])
def report(M,v):
 c=fit(M,v); r=v-M@c; h=(M*np.linalg.pinv(M).T).sum(1); return np.sqrt(np.mean(r*r)),np.median(abs(r/(1-h[:,None]))),np.max(abs(r)),c
u,v=[s['panel_a'][i] for i in ['u','v']]; rng=np.random.default_rng(1)
for kind in ['aff','cyc','poly']:
 def aa(a,i):
  p,q=a[:,i],a[:,(i+1)%4];return {'aff':aff(a),'cyc':np.c_[np.ones(len(a)),p,q,p*q,p**3],'poly':poly(a)}[kind]
 M=[aa(u,i) for i in range(4)]; p=M[0].shape[1]; best=(np.inf,None)
 for _ in range(1500):
  ix=rng.choice(len(u),p,replace=False); C=np.stack([fit(m[ix],v[ix,i]) for i,m in enumerate(M)]); e=np.linalg.norm(v-np.column_stack([m@c for m,c in zip(M,C)]),axis=1); med=np.median(e)
  if med<best[0]:best=(med,e)
 ix=best[1]<max(.025,3*best[0]); print('A',kind,best[0],ix.sum()); C=np.stack([fit(m[ix],v[ix,i]) for i,m in enumerate(M)]); print(C.round(4));print('A residual sorted',np.sort(np.linalg.norm(v-np.column_stack([m@c for m,c in zip(M,C)]),axis=1)).round(4))
 if kind=='aff': z=np.linalg.solve(C[:,1:],(x-C[:,0]).T).T
 if kind=='cyc':
  from scipy.optimize import root
  zz=np.stack([root(lambda a: np.array([aa(a[None,:],i)[0]@c for i,c in enumerate(C)])-b,b).x for b in x])
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[panel]['u'];v=s[panel][key]; print(panel,key,'extents',v.min(0),v.max(0)); print('poly2 coef',report(poly(u),v)[3].round(3));
 for n,f in [('erf',erf),('tanh',np.tanh),('sin',np.sin),('rat',lambda a:a/(1+a*a)),('sat',lambda a:a*a/(1+a*a)),('cos',np.cos)]:
  for base in [aff,poly] if panel=='panel_b' else [aff]:
   ans=minimize_scalar(lambda w:report(np.c_[base(u),f(w*u)],v)[0],bounds=(.2,4),method='bounded');print(n,base.__name__,round(ans.x,4),np.round(report(np.c_[base(u),f(ans.x*u)],v)[:3],5))
 print('u,v sample',np.c_[u,v][:12].round(3))
np.savez('/tmp/probe.npz',z=z,zz=zz)
