import pickle,importlib.util,glob
import numpy as np
from scipy.optimize import least_squares
p=glob.glob('/tmp/hegel_continual_archive/*_solver.py')[0]; s=importlib.util.spec_from_file_location('archive',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; te=d['test']; dg=i['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; out=m.solve_batch(d); tau,w=(out[k] for k in ('estimated_taus','hereditary_weights')); print('train spectrum',tau,w)
xd=m._features(tr,dg)[1]; spec=m._specimen(dg,xd,m._axial_load_map(d['calibration_load_cells'])); c,r=m._branches(spec,dt,t,tau); y=dg['strain_curves_obs']-dg['strain_0'][:,None]; v=dg['instant_rate_obs']; base=np.c_[c@w[:2]+(spec[2]@w[2:])[:,None]*t,r@w[:2]+spec[2]@w[2:]]; yy=np.c_[y,v]; print('baseline residual RMSE strain/rate',np.sqrt(np.mean((yy-base)**2,axis=0)).round(5));
def decomp(ts,ret=False):
 A=np.r_[np.c_[-np.expm1(-t[:,None]/ts),t],np.r_[1/ts,1][None,:]]; coeff=np.linalg.lstsq(A,yy.T,rcond=None)[0].T; err=coeff@A.T-yy; return (coeff,err) if ret else err.ravel()
for ts in (tau,np.exp(least_squares(lambda q:decomp(np.exp(q)),np.log(tau)).x)):
 coef,err=decomp(ts,True); print('diag spectrum',ts,'rmse',np.sqrt(np.mean(err**2,0)).round(6),'max',abs(err).max()); print('effective coeff mean/sd',coef.mean(0),coef.std(0)); print('coeff correlation',np.corrcoef(np.c_[coef,r*tau*w[:2],spec[2]@w[2:],spec[1],spec[0]].T)[:3].round(3));print('coef example rows',np.c_[coef,r*tau*w[:2],spec[2]@w[2:]][:6].round(5))
