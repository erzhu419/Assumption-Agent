from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pickle, sys
import numpy as np
root=Path('/holdout_task')
spec=spec_from_file_location('candidate',sys.argv[1] if len(sys.argv)>1 else root/'batch_solver.py')
m=module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.loads((root/'public_batch.pkl').read_bytes())[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; a=np.arange(len(x))
r=np.sum(x*x,axis=1)
slices=[(f'block{i}',j) for i,j in enumerate(np.array_split(a,6))]+[('high_radius',np.argsort(r)[-12:]),('low_radius',np.argsort(r)[:12]),('requested_tail',a[56:])]
rows=[]
for name,j in slices:
 i=np.setdiff1d(a,j); out=m.solve_batch(dict(s,train={'x':x[i],'y':y[i],'traj':z[i]},test={'x':x[j]})); e=np.abs(out['state_pred']-z[j]); ey=np.abs(out['y_hat']-y[j]); vals=[ey.mean(),e.mean(),e[:,-1].mean(),e.mean(0).max()]; rows.append(vals); print(name,np.round(vals,6))
print('block avg',np.round(np.mean(rows[:6],0),6),'worst',np.round(np.max(rows,0),6));
