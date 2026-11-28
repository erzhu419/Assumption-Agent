import importlib.util, pickle, sys
import numpy as np
from sklearn.model_selection import KFold
p=sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('cand',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=[tr[k] for k in ['x','y','traj']]; n=len(x)
folds=[('52/20',np.arange(52),np.arange(52,n))]+[(f'kfold{seed}-{i}',a,b) for seed in (217,481,14) for i,(a,b) in enumerate(KFold(4,shuffle=True,random_state=seed).split(x))]
for j in range(4):
 o=np.argsort(x[:,j]); folds.extend([(f'low{j}',o[12:],o[:12]),(f'high{j}',o[:-12],o[-12:])])
o=np.argsort(np.linalg.norm(x,axis=1));folds.append(('large radius',o[:-14],o[-14:]))
for lab,a,b in folds:
 out=m.solve_batch({**s,'train':{k:tr[k][a] for k in ('x','y','traj')},'test':{'x':x[b]}}); ey=abs(out['y_hat']-y[b]); et=abs(out['state_pred']-t[b]); print(f'{lab:14s} {ey.mean():.6f} {et.mean():.6f} {et[:,-1].mean():.6f} y_max={ey.max():.6f}')
