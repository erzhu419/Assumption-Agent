import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']
def D(x,d):
 return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def fit(a,b,n):
 w=np.linalg.lstsq(a[:n],b[:n],rcond=None)[0]; return np.mean(abs(a[:n]@w-b[:n])),np.mean(abs(a[n:]@w-b[n:]))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k,'n',len(u))
 for name,v in p.items():
  if name=='u': continue
  print(name,'poly', [fit(D(u,d),v,len(u)*3//4) for d in (1,2,3)],'delta sv',np.linalg.svd(v-u,compute_uv=False)); print('affine',np.round(np.linalg.lstsq(D(u,1),v,rcond=None)[0],3))
print('x covariance', np.round(np.cov(x.T),3)); print('tr sv',np.linalg.svd(tr,compute_uv=False))
params=[]
for z in tr:
 def obj(r,ret=False):
  h=-np.expm1(-r*t); a=(z@h)/(h@h); return (a,r,np.mean(abs(a*h-z))) if ret else np.sum((a*h-z)**2)
 rr=minimize_scalar(obj,bounds=(0.03,5),method='bounded').x; params.append(obj(rr,1))
p=np.array(params); print('a,rate,resid percentiles',np.percentile(p,[0,10,50,90,100],axis=0)); print(np.round(np.c_[x,y,p][:20],4));
for d in (1,2,3,4):
 a=D(x,d); print('degree',d,[(name,fit(a,z,52)) for name,z in [('y',y),('tr',tr),('amp',p[:,0]),('rate',p[:,1])]]);
 if d<=2: print('coef y amp and rate',np.round(np.linalg.lstsq(a,np.c_[y,p[:,:2]],rcond=None)[0],3))
print('amplitude correlations',np.corrcoef(np.c_[x,y,p[:,:2]].T).round(2))
