import sys, pickle, itertools, json
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); z=s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']; r=s['nuisance_ridge']; rho=s['source_ridge']; pp=m._fit_calibration(*m._sweeps(s)); print('params',pp.round(7)); print('roundtrip',np.max(abs(m._readout(y.reshape(-1,4),pp).ravel()-s['y_obs_raw'])))
x,v,b,t,u=m._sweeps(s); ex=np.zeros_like(v); es=np.zeros_like(u); ax=np.eye(4)[b]*t[:,None]; gi=np.arange(len(x)); si=np.arange(len(b));
for f in range(4):
 a=gi%4!=f; d=si%4!=f; p=m._fit_calibration(x[a],v[a],b[d],t[d],u[d]); ex[~a]=m._readout(x[~a],p)-v[~a]; es[~d]=m._readout(ax[~d],p)[np.arange(sum(~d)), b[~d]]-u[~d]
for name,e in [('axis',ex[np.count_nonzero(x,axis=1)==1]),('mixed',ex[np.count_nonzero(x,axis=1)>1]),('single',es)]: print('cal',name,np.sqrt(np.mean(e[abs(e)<.005]**2)),np.max(abs(e[abs(e)<.005])),sum((abs(e)>.005).ravel()))
ns=len(s['sensor_ij']); ii=np.arange(len(y)); folds=[(ii%ns)//4%4,(ii//ns),ii%4]
for walls in itertools.product(['open','reflect','periodic'],repeat=2):
 S,U=m._forward(s,walls)
 def fit(train,test,rho,r):
  a,z0,y0=S[train],z[train],y[train]; D=np.c_[a,y0]; D-=z0 @ np.linalg.solve(z0.T@z0+r*np.eye(z.shape[1]),z0.T@D); w,nu=m._qp(D[:,:-1],D[:,-1],rho,cap,budget); g=np.linalg.solve(z0.T@z0+r*np.eye(z.shape[1]),z0.T@(y0-a@w)); return w,S[test]@w+z[test]@g-y[test]
 b0=np.ones(len(y),bool); w,res=fit(b0,b0,rho,r); wu,ru=fit(b0,b0,1e-10,1e-10); print(walls,'training',np.linalg.norm(res).round(7),np.linalg.norm(ru).round(7),'weights',w.round(4))
 for fold in folds:
  print('  cv', [round(np.sqrt(np.mean(np.concatenate([fit(fold!=i,fold==i,q,k)[1] for i in np.unique(fold)])**2)),7) for q,k in [(rho,r),(1e-10,1e-10)]])
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('log',e['completed_campaign_short_name'],np.mean(e['facility_telemetry_trace'],axis=0)[[3,7]].round(4))
