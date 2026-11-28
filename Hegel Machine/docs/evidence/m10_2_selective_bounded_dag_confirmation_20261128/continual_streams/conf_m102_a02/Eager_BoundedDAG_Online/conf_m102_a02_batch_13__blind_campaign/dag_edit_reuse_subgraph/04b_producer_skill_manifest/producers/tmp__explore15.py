import numpy as np,pickle,sys,importlib.util
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
print('keys',s.keys()); x=s['train']['x']; y=np.column_stack([s['train']['y'],s['train']['traj']]); print('sv y',np.linalg.svd(y,compute_uv=False).round(4))
def poly(a,d):
 return np.column_stack([np.prod(a[:,p],axis=1) for k in range(d+1) for p in combinations_with_replacement(range(a.shape[1]),k)])
def cv(q):
 e=[]
 for v in [np.arange(52,72),*np.array_split(np.random.default_rng(9).permutation(len(q)),4)]:
  t=np.setdiff1d(np.arange(len(q)),v); r=q[v]@np.linalg.lstsq(q[t],y[t],rcond=1e-8)[0]-y[v];e.append([abs(r[:,0]).mean(),abs(r[:,1:]).mean()])
 return np.round(e,4).tolist()
for n in 'abc':
 p=s['panel_'+n]; u=p['u']; print('panel',n,'ranges',np.max(abs(u),axis=0));
 for k,v in p.items():
  if k=='u':continue
  for d in range(1,4):
   q=poly(u,d); c=np.linalg.lstsq(q,v,rcond=None)[0]; r=v-q@c; h=np.sum(np.linalg.svd(q,full_matrices=False)[0][:,:np.linalg.matrix_rank(q)]**2,axis=1); print(k,d,'r',np.percentile(np.linalg.norm(r,axis=1),[0,25,50,75,90,100]).round(5),'loo',np.median(np.linalg.norm(r,axis=1)/np.maximum(1-h,1e-8)).round(5));
   if d==1:print('aff',c.round(3))
for d in range(1,4): print('raw',d,cv(poly(x,d)))
p='/tmp/hegel_continual_archive/conf_m102_a02_batch_06_solver.py'; spec=importlib.util.spec_from_file_location('arc06',p); a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
for n in 'ab':
 z=a._calibrate(s['panel_'+n],x)
 for d in range(1,4):print('arc cycle',n,d,cv(poly(z,d)))
