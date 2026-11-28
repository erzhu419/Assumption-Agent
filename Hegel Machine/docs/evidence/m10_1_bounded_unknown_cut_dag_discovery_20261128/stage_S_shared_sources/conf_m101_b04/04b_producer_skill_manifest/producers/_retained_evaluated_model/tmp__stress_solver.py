import importlib.util
import pickle
import sys
import numpy as np
p=sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,z=(st['train'][k] for k in ('x','y','traj'))
def score(idx):
    tr=np.setdiff1d(np.arange(len(x)),idx)
    subset=dict(st,train={'x':x[tr],'y':y[tr],'traj':z[tr]},test={'x':x[idx]})
    q=m.solve_batch(subset)
    a=np.abs(q['y_hat']-y[idx]); b=np.abs(q['state_pred']-z[idx])
    return [a.mean(),b.mean(),b.mean(0).max()]
rng=np.random.default_rng(395)
slices={'fixed':np.arange(52,72),'radial':np.argsort(np.linalg.norm(x,axis=1))[-18:]}
for j in range(4):
    order=np.argsort(x[:,j]); slices[f'low{j}']=order[:18]; slices[f'high{j}']=order[-18:]
for label,idx in slices.items(): print(label,np.round(score(idx),6))
r=np.array([score(idx) for _ in range(4) for idx in np.array_split(rng.permutation(len(x)),4)])
print('random mean',r.mean(0),'random worst',r.max(0))
