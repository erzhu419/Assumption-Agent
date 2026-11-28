import numpy as np,pickle,runpy
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; f=runpy.run_path('/holdout_task/batch_solver.py')['solve_batch']; rng=np.random.default_rng(1642)
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=44).split(x))+[(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(4))]
for i,o in splits:
 y,t=tr['y'][i],tr['traj'][i]; p=f({**s,'train':{'x':x[i],'y':y,'traj':t,'summary':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])},'test':{'x':x[o]}}); print(np.round(np.r_[np.mean(abs(p['y_hat']-tr['y'][o])),np.mean(abs(p['state_pred']-tr['traj'][o]),0)],6),flush=True)
