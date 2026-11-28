import sys, pickle, numpy as np
from itertools import combinations_with_replacement as comb
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=np.vstack([t['x'],s['test']['x']]); y=t['y']; z=t['traj']; n=len(y)
np.set_printoptions(precision=5,suppress=True,linewidth=130)
a=lambda q:np.column_stack([np.ones(len(q)),q]); fit=lambda u,v:np.linalg.lstsq(a(u),v,rcond=None)[0]
print('times',s['eval_times'],'telemetry',s['telemetry_vec'],'summary',t['summary']); print('y/z ranges', [np.min(y),np.max(y)],z.min(0),z.max(0)); print('trajectory singular values and temporal basis',np.linalg.svd(z,full_matrices=False)[1:]); print('first x, y, z',x[:5],y[:5],z[:5]); print('x stats',x.mean(0),x.std(0),x.min(0),x.max(0))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k,'u mean/std',u.mean(0),u.std(0));
 for key in set(p)-{'u'}:
  v=p[key]; w=fit(u,v); f=a(u)
  for _ in range(10):
   idx=np.argsort(np.linalg.norm(f@w-v,axis=1))[:int(.82*len(u))]; w=np.linalg.lstsq(f[idx],v[idx],rcond=None)[0]
  print(key,'robust affine',w,'abs residual percentiles',np.quantile(abs(f@w-v), [0,.5,.75,.9,1]))
features=run_node({'__input__':s}); features['x']=a(x)
Y=np.column_stack([y,z]); splits=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(11).permutation(n),4)]
for name,F in features.items():
 for d in ([1] if name=='x' else [1,2,3]+([4] if name=='hybrid' else [])):
  g=F[:,:{1:5,2:15,3:35,4:70}[d]]
  vals=[]
  for reg in [1e-8,1e-4,.001,.01,.1,1]:
   ms=[]
   for i,j in splits:
    p=g[j]@np.linalg.solve(g[i].T@g[i]+reg*np.eye(g.shape[1]),g[i].T@Y[i]); e=abs(p-Y[j]); ms.append([e[:,0].mean(),e[:,1:].mean()])
   vals.append((reg,np.round(ms[0],5),np.round(np.max(ms,axis=0),5)))
  print(name,d,vals)
