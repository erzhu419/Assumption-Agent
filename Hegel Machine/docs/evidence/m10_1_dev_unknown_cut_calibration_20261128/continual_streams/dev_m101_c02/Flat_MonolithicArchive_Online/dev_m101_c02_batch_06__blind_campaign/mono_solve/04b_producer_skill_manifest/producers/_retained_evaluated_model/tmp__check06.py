import pickle, sys
import numpy as np
from sklearn.model_selection import KFold
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=s['train']; y,T=q['y'],q['traj']; w=m._calibrate(s['panel_a'],q['x']); p=s['panel_b']; u,v=p['u'],p['v']; n=len(w); folds=[(np.arange(56),np.arange(56,n)),*KFold(4,shuffle=True,random_state=4).split(w)]
for nh in (1,2):
 z=m._operator(p,np.r_[w,u],nh); print('harmonics',nh,'panel rms',np.std(z[n:]-v),flush=True); z=z[:n]
 for sparse in (True,False):
  for kind in ('linear','atan','tanh','sin','fixed'):
   es=[]
   for i,j in folds:
    tr=np.isin(np.arange(n),i); yp,tp=m._predict(w,z,tr,{k:q[k][i] for k in ('y','traj')},sparse,kind); es.append([np.mean(abs(yp-y[j])),np.mean(abs(tp-T[j])),np.mean(abs(tp[:,-1]-T[j,-1]))]);
   print(sparse,kind,np.round(es[0],6),np.round(np.max(es,axis=0),6),np.round(np.mean(es,axis=0),6),flush=True)
 D=m._design(w,z,'trajectory'); co=m._ls(D,T); print('time coefficients',np.round(co,5),'rms',np.std(D@co-T),flush=True)
