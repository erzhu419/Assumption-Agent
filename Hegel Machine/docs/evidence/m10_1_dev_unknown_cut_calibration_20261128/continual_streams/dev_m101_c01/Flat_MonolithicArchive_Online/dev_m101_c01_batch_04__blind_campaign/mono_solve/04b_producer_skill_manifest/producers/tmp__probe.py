import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from numpy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; N=len(x)
fit=lambda a,b: lstsq(a,b,rcond=1e-10)[0]
rm=lambda a: np.sqrt(np.mean(a*a))
def poly(x,d):
 return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def cv(a,b):
 c=fit(a[:56],b[:56]); r=a@c-b; return np.mean(abs(r[:56])),np.mean(abs(r[56:]))
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; mask=np.max(np.abs(v),axis=1)<2
 a=poly(u[mask],1); c=fit(a,v[mask]); print(p,k,mask.sum(),'affine',c.round(4).tolist(),'rms',rm(a@c-v[mask]));
 for f in [np.tanh, lambda z: np.sin(z),lambda z:z/(1+abs(z))]:
  out=[]
  for i in range(4):
   b=v[mask,i]; sig=np.max(abs(b)); res=least_squares(lambda w:w[0]+w[1]*f(a@w[2:])-b,np.r_[0,max(1,sig),c[:,i]],max_nfev=1000); out.append(rm(res.fun));
  print('nonlinear',out)
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']); print('svd',np.linalg.svd(z,compute_uv=False))
def temporal(r): return -np.expm1(-t[:,None]*r)
for guess in [[1.],[1.5,.2],[1.5,.5,.1]]:
 opt=least_squares(lambda r:(temporal(r)@fit(temporal(r),z.T)-z.T).ravel(),guess,bounds=(.001,9),max_nfev=1000); f=temporal(opt.x); a=fit(f,z.T).T
 print('rates',opt.x,'noise',rm(opt.fun),'coef span',a.min(0),a.max(0),'coef svd',np.linalg.svd(a,compute_uv=False));
 for d in (1,2,3):
  b=poly(x,d); print('pol',d,'state',cv(b,z),'y',cv(b,y),'y+latent',cv(np.c_[b,a],y))
q=[]
for row in z:
 r=least_squares(lambda w:temporal([w[0]])[:,0]*w[1]-row,[1,row[-1]]); q.append([*r.x,rm(r.fun)])
q=np.array(q); print('row rates',np.quantile(q,[0,.1,.5,.9,1],axis=0)); print('y corr',np.corrcoef(np.c_[y,x,z[:,-1],q[:,:2]].T)[0]);
