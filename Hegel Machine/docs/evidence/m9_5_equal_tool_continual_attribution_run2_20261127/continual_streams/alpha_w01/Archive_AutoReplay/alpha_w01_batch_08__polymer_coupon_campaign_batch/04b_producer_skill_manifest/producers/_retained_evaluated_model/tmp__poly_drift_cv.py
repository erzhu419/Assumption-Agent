import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import StratifiedKFold
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_batch_08_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
o=pickle.load(open('/holdout_task/public_alpha_w01_batch_08.pkl','rb'))[0]; d=o['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; ys=d['yield_stress']; x,xt=m._alignment(tr,te); maps=m._load_cells(d['calibration_load_cells']); inp=m._inputs(tr,x,maps,dt,ys); keep=np.isfinite(tr['coupon_features_raw']).all(1); ids=np.flatnonzero(keep); lab=tr['lab_id']; strata=tr['stratum']; cout=np.zeros_like(tr['strain_curves_obs']); rout=np.zeros(len(x)); rm=lambda a: np.sqrt(np.mean(a*a)); mad=lambda a: 1.4826*np.median(abs(a-np.median(a)))
for a,b in StratifiedKFold(n_splits=5,shuffle=True,random_state=183).split(ids,3*lab[ids]+strata[ids]):
 a,b=ids[a],ids[b]; fit=m._fit({k:tr[k][a] for k in ('strain_curves_obs','strain_0','instant_rate_obs')},tuple(z[a] for z in inp),dt,t); c,r=m._design(fit[0],tuple(z[b] for z in inp),dt,t); cout[b]=tr['strain_curves_obs'][b]-tr['strain_0'][b,None]-c@fit[1]; rout[b]=tr['instant_rate_obs'][b]-r@fit[1]
# Robustly summarize each held-out coupon's late-horizon residual, where the
# time slope is identifiable despite isolated contaminated curve readouts.
slope=np.median(cout[:,3:]/t[3:],axis=1)
for c in range(4):
 k=keep&(lab==c); z=slope[k]; rr=rout[k]; xx=np.column_stack((np.ones(k.sum()),x[k])); sig=mad(z); p=least_squares(lambda b:(xx@b-z)/max(sig,1e-8),np.zeros(xx.shape[1]),loss='soft_l1',f_scale=1.5).x; print('lab',c,'n',k.sum(),'drift MAD',round(sig,6),'rate MAD',round(mad(rr),6),'rate minus drift MAD',round(mad(rr-z),6),'clip covariance',round(float(np.cov(np.clip(z,-3*sig,3*sig),np.clip(rr,-3*mad(rr),3*mad(rr)))[0,1]),8),'median drift',round(float(np.median(z)),6),'feature correction',np.round(p,5),'drift MAD after',round(mad(z-xx@p),6))
 print('late residual cross-cov/time',np.round(np.cov(np.clip(cout[k,3:]/t[3:],-.03,.03).T)[np.triu_indices(len(t)-3,1)].reshape(-1),6))
