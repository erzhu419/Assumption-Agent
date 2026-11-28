import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=s['train']['y'];Z=s['train']['traj'];T=np.linalg.svd(Z,full_matrices=False)[2][:2]; targets=np.c_[Y,Z@T.T]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
def panel(u,v,fun):
 f=fun(u);keep=np.ones(len(u),bool)
 for j in range(5):
  b=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0];e=np.linalg.norm(f@b-v,axis=1);keep=e<3
 return b,e
from itertools import product
a=s['panel_a'];u,v=a['u'],a['v']; good=np.linalg.norm(v,axis=1)<7
p=poly(u,3); cf=np.linalg.lstsq(p[good],v[good],rcond=None)[0]; print('panel A inliers',sum(good),'max residual',np.max(abs(p[good]@cf-v[good])),'latent bounds',u.min(0),u.max(0)); print('coeff',np.round(cf,5))
lin=cf[1:5]; z=[]
for obs in x:
 r=least_squares(lambda q:poly(q[None],3)[0]@cf-obs,(obs-cf[0])@np.linalg.inv(lin),gtol=1e-12,xtol=1e-12); z.append(r.x)
z=np.array(z);print('inverse ranges',z.min(0),z.max(0),'rms',np.mean((poly(z,3)@cf-x)**2)**.5)
c=s['panel_c']; cfs={}
for k in ['v1','v2']:
 for d in [1,2,3]:
  p=poly(c['u'],d); f=np.linalg.lstsq(p,c[k],rcond=None)[0]; e=p@f-c[k]; lev=np.sum(p*np.linalg.pinv(p).T,axis=1); print('C',k,d,p.shape,'rmse',np.mean(e*e)**.5,'loo',np.mean((e/np.maximum((1-lev)[:,None],1e-9))**2)**.5)
  cfs[k,d]=f
b=s['panel_b'];print('panel b head',np.round(b['u'][:5],3),np.round(b['v'][:5],3))
np.savez('/tmp/cubic_probe.npz',z=z,cf=cf)
res=[]; idx=np.arange(52); te=np.arange(52,72)
def score(A,tag):
 U,d,V=np.linalg.svd(A[idx],full_matrices=False); m=U.T@targets[idx]; B=A[te]@V.T
 for l in [1e-9,1e-6,.0001,.001,.003,.01,.03,.1,.3,1]:
  pred=B@((d/(d*d+l))[:,None]*m);res.append((np.mean(abs(pred[:,0]-Y[te])),np.mean(abs((pred[:,1:]-targets[te,1:])@T)),tag,l))
for key,w in [('axis',z),('raw',x)]+[(str(k),poly(z,k[1])@f) for k,f in cfs.items()]:
 for sq in [0,.09,.22,.5,1.25]:
  h=w/np.sqrt(1+sq*w*w)
  for d in [2,3,4]:score(poly(h,d),(key,sq,'poly',d))
  for f in [.75,1,1.25,1.5,1.75,2,2.15,2.25,2.5,3]:
   for d in [1,2,3]: score(np.c_[poly(h,d),np.sin(f*h),np.cos(f*h)],(key,sq,'four',f,d))
for j in [0,1]: print('best',j,*sorted(res,key=lambda r:r[j])[:8],sep='\n')
