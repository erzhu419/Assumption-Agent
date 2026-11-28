import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from importlib.util import spec_from_file_location,module_from_spec
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; x,y,T=map(tr.get,('x','y','traj'))
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda a:np.c_[np.ones(len(a)),a]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for n in range(1,d+1) for k in combinations_with_replacement(range(x.shape[1]),n)])
for p in ['panel_a','panel_b','panel_c']:
 d=s[p]; u=d['u']; print(p)
 for k,v in d.items():
  if k=='u':continue
  for n in [1,2,3]:
   A=poly(u,n); ii=np.arange(len(u));
   if p=='panel_a':
    rng=np.random.default_rng(3); best=None
    for _ in range(500):
     j=rng.choice(len(u),A.shape[1],replace=False); e=np.linalg.norm(v-A@ls(A[j],v[j]),axis=1); score=np.median(e)
     if best is None or score<best[0]:best=score,e
    ii=np.argsort(best[1])[:45]
   b=ls(A[ii],v[ii]); e=v-A@b; print(k,n, np.sqrt(np.mean(e[ii]**2)), np.sort(np.linalg.norm(e,axis=1))[-4:], 'coef' if n==1 else '', b.round(4) if n==1 else '')
print('traj sv',np.linalg.svd(T,compute_uv=False)); print('centered sv',np.linalg.svd(T-T.mean(0),compute_uv=False)); print('mean/std',T.mean(0),T.std(0),'corr y',np.corrcoef(np.c_[y,T].T)[0]);
for name in ['00','01']:
 spec=spec_from_file_location('m',f'/tmp/hegel_continual_archive/dev_m101_c04_batch_{name}_solver.py'); m=module_from_spec(spec); spec.loader.exec_module(m)
 for j in range(3):
  i=np.arange(72) if j==0 else np.random.default_rng(j).permutation(72); a,b=i[:56],i[56:]; out=m.solve_batch(dict(s,train={k:v[a] if k!='summary' else v for k,v in tr.items()},test={'x':x[b]})); ey=out['y_hat']-y[b]; et=out['state_pred']-T[b]; print(name,j,np.mean(abs(ey)),np.mean(abs(et)),np.mean(abs(et),axis=0))
