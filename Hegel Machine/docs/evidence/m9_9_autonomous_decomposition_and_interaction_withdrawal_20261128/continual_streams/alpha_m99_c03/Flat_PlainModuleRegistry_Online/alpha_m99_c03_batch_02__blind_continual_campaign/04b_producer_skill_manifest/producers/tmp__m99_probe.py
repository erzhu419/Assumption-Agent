import sys,pickle,json,importlib.util
import numpy as np
from itertools import combinations_with_replacement
p='/tmp/hegel_continual_archive/alpha_m99_c03_s00_solver.py'; s=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; te=d['test']; g=i['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; cal=d['calibration_load_cells']; x=cal['true_channel_loads']; y=cal['bridge_raw']; np.set_printoptions(precision=5,suppress=True)
def poly(x, degree):
 terms=[()] + [z for k in range(1,degree+1) for z in combinations_with_replacement(range(4),k)]
 return np.column_stack([np.prod(x[:,z],axis=1) for z in terms]),terms
if __name__=='__main__':
 for k in range(1,4):
  A,terms=poly(x,k); B=np.linalg.lstsq(A,y,rcond=None)[0]; r=y-A@B; h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]**2,axis=1); loo=r/np.maximum(1-h[:,None],1e-12)
  print('bridge',k,'rank',np.linalg.matrix_rank(A),'RMS',np.sqrt(np.mean(r*r,axis=0)),'LOO RMS',np.sqrt(np.mean(loo*loo,axis=0))); print('large coeffs',[(z,b.round(4).tolist()) for z,b in zip(terms,B) if np.max(abs(b))>.004])
 for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
  a=np.asarray(e['facility_telemetry_trace']); print('apparatus', {k:v for k,v in e.items() if k!='facility_telemetry_trace'},'mean',a.mean(0),'std',a.std(0))
 bridge=m._calibration(cal); X=m._features(tr,te); n=len(tr['lab_id']); ct=m._coupon_inputs(tr,X[:n],bridge); taus,theta=m._fit(ct,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('archived fit',taus,theta)
 Xg=m._features(tr,g)[n:]; cg=m._coupon_inputs(g,Xg,bridge); T=np.c_[-np.expm1(-t[:,None]/taus),t]; print('time basis condition',np.linalg.cond(T))
 for name,b,c in [('train',tr,ct),('diag',g,cg)]:
  F,R=m._design(taus,c,dt,t); obs=b['strain_curves_obs']-b['strain_0'][:,None]; rate=b['instant_rate_obs']; err=obs-F@theta; er=rate-R@theta; raw_amp=np.linalg.lstsq(T,obs.T,rcond=None)[0].T; gap=R[:,:2]*taus; amp0=np.c_[gap*theta[:2],R[:,2:]@theta[2:]]; ej=raw_amp-amp0
  print(name,'resid time med',np.median(err,axis=0),'rmse',np.sqrt(np.mean(err**2,axis=0)),'rate median rmse',np.median(er),np.sqrt(np.mean(er**2)),'amp correction med/std',np.median(ej,axis=0),np.std(ej,axis=0),'basis error RMS',np.sqrt(np.mean((obs-raw_amp@T.T)**2))); print('examples hold, gaps, drift, rel amp, rate correction',np.c_[c[1],gap,amp0[:,2],ej,er][:10])
