import sys,importlib.util,pickle,numpy as np
from sklearn.model_selection import KFold

def load(path):
 spec=importlib.util.spec_from_file_location('candidate',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];n=len(tr['x']);i=np.arange(n)
base=load('/tmp/baseline.py'); w=base._operator(s,tr['x']); rad=np.linalg.norm(w,axis=1)
splits=[('prescribed',i[:52],i[52:])]+[(f'kfold{j}',a,b) for j,(a,b) in enumerate(KFold(6,shuffle=True,random_state=82).split(i))]
for label,idx in [('radius',rad)]+[(f'axis{j}',w[:,j]) for j in range(4)]:
 order=np.argsort(idx)
 for side,b in [('high',order[-12:]),('low',order[:12])]:splits.append((label+side,np.setdiff1d(i,b),b))
for path in sys.argv[1:]:
 m=load(path); print(path); scores=[]
 for l,a,b in splits:
  tt={k:tr[k][a] for k in ('x','y','traj')};t=tt['traj'][:,-1];tt['summary']=np.array([tt['y'].mean(),tt['y'].std(),t.mean(),t.std()]); o=m.solve_batch({**s,'train':tt,'test':{'x':tr['x'][b]}}); dy=abs(o['y_hat']-tr['y'][b]);dt=abs(o['state_pred']-tr['traj'][b]); z=[dy.mean(),dt.mean(),dt[:,-1].mean(),dt.mean(0).max()]; print(l,np.round(z,6));scores.append(z)
 print('worst',np.max(scores,axis=0))
