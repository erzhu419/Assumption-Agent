import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def poly(u,order):
 return np.column_stack([np.ones(len(u))]+[np.prod(u[:,j],axis=1) for d in range(1,order+1) for j in combinations_with_replacement(range(u.shape[1]),d)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def loo(a,b):
 q=a@np.linalg.pinv(a);return (b-q@b)/np.maximum(1e-8,1-np.diag(q))[:,None]
for name in ['panel_a','panel_b','panel_c']:
 p=s[name];u=p['u'];
 for k,b in p.items():
  if k=='u':continue
  print(name,k,'ranges',u.min(0),u.max(0),b.min(0),b.max(0));print('affine',fit(poly(u,1),b));
  for d in range(1,4 if len(u)>35 else 3):
   r=loo(poly(u,d),b);print(d,'abs quantiles',np.quantile(abs(r),[.5,.8,.95,1],axis=0));
  for shift in (1,2,3):
   rs=[];cs=[]
   for j in range(4):
    x,y=u[:,j],u[:,(j+shift)%4]; a=np.column_stack([x*0+1,x,y,x*y,x**3,y**3,x*x,y*y]); rs.append(loo(a,b[:,j:j+1])[:,0]);cs.append(fit(a,b[:,j]));
   print('pair',shift,'coef',np.array(cs),'L1 quantiles',np.quantile(np.abs(rs),[.5,.8,.95,1],axis=1))
x=s['train']['x']; z=np.column_stack([s['train']['y'],s['train']['traj']]);print('traj SV',np.linalg.svd(z[:,1:],compute_uv=False));print('raw x range',x.min(0),x.max(0));
for d in (1,2,3):
 a=poly(x,d);e=abs(a[52:]@fit(a[:52],z[:52])-z[52:]); print('raw degree',d,e.mean(0));
print('times,telemetry',s['eval_times'],s['telemetry_vec'])
