import numpy as np, runpy, pickle
from itertools import combinations
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
N=runpy.run_path('/holdout_task/node_01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=N['_calibrate'](s['panel_a'],t['x']); Y=np.c_[t['y'],t['traj']]; n=len(x); p=list(combinations(range(x.shape[1]),2)); P=np.c_[[x[:,i]+x[:,j] for i,j in p]+[x[:,i]-x[:,j] for i,j in p]].T

def f(w,kind):
 q=np.c_[x] if kind=='add' else np.c_[x,2*x,P]; h=N['_harmonics'](q,w);return np.c_[np.ones(n),x,h]

def err(w,i,j,k,penalty):
 F=f(w,k);a=F[i]; g=np.linalg.solve(a.T@a+penalty*np.eye(a.shape[1]),a.T@Y[i]); e=F[j]@g-Y[j];return g,np.array([abs(e[:,0]).mean(),abs(e[:,1:]).mean()])

folds=[(i,j) for r in range(5) for i,j in KFold(6,shuffle=True,random_state=721+r).split(x)]; ii=np.arange(n)
for kind in ['add','pair']:
 for target in [0,1]:
  scores=[]; ws=[]
  for i,j in folds:
   w=minimize_scalar(lambda w:err(w,i,i,kind,.003)[1][target],bounds=(2.15,3),method='bounded').x
   ws.append(w);scores.append(err(w,i,j,kind,.003)[1][target])
  w=minimize_scalar(lambda w:err(w,ii,ii,kind,.003)[1][target],bounds=(2.15,3),method='bounded').x
  g,e=err(w,ii,ii,kind,.003); print(kind,target,'frequency',w,'fold range',min(ws),max(ws),'fit error',e[target],'cv mean/max',np.mean(scores),max(scores)); print('coefficients',np.round(g[:,0] if target==0 else g[:,-1],4))
