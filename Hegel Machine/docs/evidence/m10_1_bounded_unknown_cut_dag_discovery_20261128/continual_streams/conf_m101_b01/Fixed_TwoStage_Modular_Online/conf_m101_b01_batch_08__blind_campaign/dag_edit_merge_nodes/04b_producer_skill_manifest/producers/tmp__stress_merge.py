import os,pickle,runpy,numpy as np
from sklearn.model_selection import KFold
n=runpy.run_path('/holdout_task/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['x']); splits=[(np.arange(N-20),np.arange(N-20,N))]+list(KFold(4,shuffle=True,random_state=17).split(tr['x']))
a=n['a_fit'](s['panel_a']); b=n['b_fit'](s['panel_b']); c,d=n['c_fit'](s['panel_c']); F=n['_features'](tr['x'],a,b,c,d)
for i,j in splits:
 pred=n['_readout']([f[np.r_[i,j]] for f in F],tr['y'][i],tr['traj'][i]); e=np.mean(abs(pred-np.c_[tr['y'][j],tr['traj'][j]]),axis=0); print('stress baseline',np.round([e[0],e[1:].mean(),e[1:].max()],5))
