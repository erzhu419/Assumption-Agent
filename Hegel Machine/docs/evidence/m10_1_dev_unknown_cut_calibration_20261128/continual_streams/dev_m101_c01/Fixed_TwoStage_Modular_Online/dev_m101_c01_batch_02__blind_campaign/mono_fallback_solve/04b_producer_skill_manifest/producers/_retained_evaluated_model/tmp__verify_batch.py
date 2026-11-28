import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import _components
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study=pickle.load(f)[0]['study']
x,y,z=[study['train'][k] for k in ('x','y','traj')]
rows=np.arange(len(x)); radius=np.linalg.norm(x,axis=1)
folds=[('ordered',rows[56:]), *[(f'shuffle{i}',j) for i,j in enumerate(np.array_split(np.random.default_rng(22).permutation(len(x)),5))], ('outer-radius',np.argsort(radius)[-14:]), *[(f'axis{i}',np.argsort(abs(x[:,i]))[-14:]) for i in range(4)]]
for name,j in folds:
    i=np.setdiff1d(rows,j)
    a,b=_components(x[i],y[i],z[i],x[j]); p=a@np.array([.25,.30,.30,.15]); q=np.average(b,axis=0,weights=[.4,.2,.4]); print(name,'blended', np.round([np.abs(p-y[j]).mean(),np.abs(q-z[j]).mean(),np.abs(q[:,-1]-z[j,-1]).mean()],6),'components',np.abs(a-y[j,None]).mean(0).round(5),np.abs(b-z[j]).mean((1,2)).round(5),flush=True)
