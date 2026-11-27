import importlib.util as iu, pickle, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
mods=[]
for p in sorted(Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 spec=iu.spec_from_file_location(p.stem,p); m=iu.module_from_spec(spec); spec.loader.exec_module(m); mods.append(m)
b=s['sensor_calibration_sweeps']; r,y=b['ref_rosette'],b['raw_rosette']; a=b['single_bank_sweeps']; print('obs range',s['y_obs_raw'].min(),s['y_obs_raw'].max()); print('aux correlations',np.corrcoef(s['aux_manifold_telemetry'].T).round(3))
for j in range(4):
 idx=a['bank_id']==j; z=np.zeros((idx.sum(),4)); z[:,j]=a['ref_signal'][idx]; h=np.r_[r,z]; v=np.r_[y[:,j],a['raw_readout'][idx]]; ok=np.abs(v)<8
 x,n,p=h[:,j],h[:,(j+1)%4],h[:,(j-1)%4]
 D=np.array([np.ones(len(h)),x,n,p,x*n,x**3,x*p,n*p,x**2,n**2,p**2,n**3,p**3,h[:,(j+2)%4],x*n*p]).T
 print('bank',j,'clean',ok.sum(),'of',len(v),'suspect',np.round(v[~ok],2))
 for k in [4,6,15]:
  d=D[:,:k]; se=[]
  for sel in np.array_split(np.where(ok)[0],5):
   tr=ok.copy(); tr[sel]=False; cc=np.linalg.lstsq(d[tr],v[tr],rcond=None)[0]; se.extend((v[sel]-d[sel]@cc)**2)
  cc=np.linalg.lstsq(d[ok],v[ok],rcond=None)[0]; print(k,'cv RMSE',np.mean(se)**.5,'coef',cc.round(5))
yc=mods[2]._calibrated_observations(s)
Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); R=lambda A:A-Z@(P@A)
for m in mods:
 try:
  S,U=m._forward(s); D=R(S); t=R(yc); ridge=s['source_ridge']; ans=m._box_simplex(D.T@D+ridge*np.eye(D.shape[1]),-D.T@t,s['u_box'],s['total_budget']); w=ans[0]; print(m.__name__,'resid',np.linalg.norm(D@w-t),'per shift',np.linalg.norm((D@w-t).reshape(3,-1),axis=1),'w',w.round(5))
 except Exception as e: print(m.__name__,repr(e))
