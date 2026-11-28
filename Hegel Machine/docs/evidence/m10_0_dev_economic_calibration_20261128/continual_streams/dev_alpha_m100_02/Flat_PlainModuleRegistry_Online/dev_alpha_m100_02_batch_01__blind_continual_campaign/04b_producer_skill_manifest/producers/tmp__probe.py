import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/dev_alpha_m100_02_s00_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,c=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; B=m['_bridge'](cal)
raw=cal['bridge_raw']; L=cal['true_channel_loads']; er=raw@B[1:]+B[0]-L
print('bridge RMS/max',np.sqrt(np.mean(er**2,axis=0)),np.max(abs(er),axis=0),'sum rms',np.std(er.sum(1)),'B',B)
sw=cal['single_channel_sweeps'];
for lab in np.unique(sw['lab_id']):
 k=sw['lab_id']==lab; print('sweep',lab,np.round(np.column_stack((sw['cmd_stress'][k],sw['true_stress'][k])),4).T)
o=m['solve_batch'](d); tau=o['estimated_taus']; w=o['hereditary_weights']; print('tau,w',tau,w)
for p in [tr,c]:
 x=m['_features'](tr,te)[0] if p is tr else m['_features'](tr,c)[1]; s=m['_prepare'](p,x,B); H,R=m['_design'](np.log(tau),s,dt,t); base=np.c_[H@w,R@w]; y=np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; e=y-base; print('RMS by t/rate',np.round(np.mean(e**2,axis=0)**.5,4),'MAE',np.round(np.mean(abs(e),axis=0),4));
 basis=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]; a=e@np.linalg.pinv(basis).T; e2=e-a@basis.T
 print('three component residual RMS',np.sqrt(np.mean(e2**2)), 'svd',np.linalg.svd(e,compute_uv=False)[:8]); print('decomposed:'); print(np.round(np.c_[p['lab_id'],s[1],R[:,:2]*tau,s[2]@w[2:],a,np.sqrt(np.mean(e2**2,axis=1))],4)[:24]);
