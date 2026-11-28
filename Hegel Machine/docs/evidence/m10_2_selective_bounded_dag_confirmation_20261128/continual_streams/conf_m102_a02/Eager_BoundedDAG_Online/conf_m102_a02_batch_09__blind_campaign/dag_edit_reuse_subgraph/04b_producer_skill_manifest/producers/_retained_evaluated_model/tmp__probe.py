import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.linalg import svdvals
np.set_printoptions(precision=5,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def design(x,d=1):return np.column_stack([np.ones(len(x))]+[x[:,p].prod(axis=1) for j in range(1,d+1) for p in cwr(range(x.shape[1]),j)])
def report(z,y):
 b=np.linalg.lstsq(z,y,rcond=None)[0]; e=y-z@b; h=np.sum(np.linalg.svd(z,full_matrices=False)[0]**2,axis=1); return np.mean(abs(e),axis=0),np.mean(abs(e/np.maximum(1-h,1e-8).reshape((-1,)+(1,)*(y.ndim-1))),axis=0),b
for pn in ['panel_a','panel_b','panel_c']:
 p=s[pn];u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  print(pn,k,'input extrema',u.min(),u.max(),'output extrema',v.min(),v.max());
  for name,z in [('linear',design(u)),('quad',design(u,2)),('cubic',design(u,3)),('diag cubic',np.c_[design(u),u**2,u**3]),('diag quintic',np.c_[design(u),u**2,u**3,u**4,u**5]),('sin',np.c_[design(u),np.sin(u),np.cos(u)])]:
   if z.shape[1]<len(z):
    err,loo,b=report(z,v);print(name,err,loo); 
    if name in ('linear','diag cubic') and pn=='panel_a': print(b)
tr=s['train'];x=tr['x']; y=np.c_[tr['y'],tr['traj']]; print('target singular values',svdvals(y[:,1:]));print('telemetry',s['telemetry_vec']);print('x extrema',x.min(),x.max());
for name,z in [('linear',design(x)),('quad',design(x,2)),('cubic',design(x,3)),('diag',np.c_[design(x,2),x**3,x**5])]:
 err,loo,b=report(z,y); print(name,err,loo); print('resid singular values',svdvals((y-z@b)[:,1:]));
 if name=='linear':print(b)
