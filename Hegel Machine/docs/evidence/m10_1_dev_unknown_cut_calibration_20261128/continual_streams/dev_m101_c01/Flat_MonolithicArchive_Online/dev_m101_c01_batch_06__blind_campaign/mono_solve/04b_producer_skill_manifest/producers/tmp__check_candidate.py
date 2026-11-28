import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _bank,_combine
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); x,y,z=(tr[k] for k in ('x','y','traj'))
rng=np.random.default_rng(706); splits=[np.arange(56,72)]+[j for _ in range(3) for j in np.array_split(rng.permutation(n),5)]+[np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-14:]]
from collections import defaultdict
errors=defaultdict(list)
for v in splits:
 i=np.setdiff1d(np.arange(n),v); target,l,p,mode=_bank(s,x[i],x[v],y[i],z[i]); p['stack']=np.c_[_combine(target,l,p,0),_combine(target,l,p,1)]
 for name,h in p.items():
  dz=np.abs(h[:,1,None]*mode-z[v]); errors[name].append([np.abs(h[:,0]-y[v]).mean(),dz.mean(),dz[:,-1].mean()])
for j in range(3):
 print('OUTPUT',j,'columns 56/16, mean ordinary, worst ordinary, outer-radius'); vals={k:np.round([a[0,j],a[:-1,j].mean(),a[:-1,j].max(),a[-1,j]],6) for k,e in errors.items() for a in [np.array(e)]};
 for k in ['stack','quadratic']+sorted(vals,key=lambda k:vals[k][1]+.3*vals[k][2]+.1*vals[k][3])[:8]:print(k, vals[k])
print('temporal residual',np.std(z-np.outer(z@mode,mode),axis=0))
