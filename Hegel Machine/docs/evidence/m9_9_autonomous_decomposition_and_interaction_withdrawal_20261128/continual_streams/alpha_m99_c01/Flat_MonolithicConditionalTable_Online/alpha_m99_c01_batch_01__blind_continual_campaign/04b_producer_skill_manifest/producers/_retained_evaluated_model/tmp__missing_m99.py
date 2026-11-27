import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr=d['train'];te=d['test'];c=i['campaign_diagnostics'];dt=d['past_dt'];t=d['eval_times'];cal=d['calibration_load_cells'];raw=cal['bridge_raw'];y=cal['true_channel_loads'];b=m._bridge(cal)
print('bridge',b,'raw range',np.min(raw,0),np.max(raw,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
from itertools import combinations_with_replacement
for degree in (1,2,3):
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,j],axis=1) for j in range(1,degree+1) for j in combinations_with_replacement(range(4),j)])
 pred=np.zeros_like(y)
 for q in np.array_split(np.random.default_rng(5).permutation(len(y)),5):
  z=np.setdiff1d(np.arange(len(y)),q);pred[q]=A[q]@np.linalg.lstsq(A[z],y[z],rcond=None)[0]
 e=pred-y;print('bridge OOF degree',degree,'channel rmse',np.sqrt(np.mean(e**2,0)),'stress rmse',np.sqrt(np.mean(e.sum(1)**2)))
x,xt=m._features(tr,te);p,h=m._loads(tr,b);pc,hc=m._loads(c,b);tau,beta=m._fit(x,p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,beta)
_,xc=m._features(dict((k,np.concatenate([tr[k],te[k]],axis=0)) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)
R=c['coupon_features_raw']; lab=c['lab_id']; st=c['stratum']; xdiag=m._features(c,c)[1]; complete=np.isfinite(R).all(1)
F=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); y=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; drift=beta[2]+xdiag@beta[3:]; base=(hc[:,None]*beta[:2])@F.T
coef=np.linalg.lstsq((drift[complete,None,None]*F[None,:,:]).reshape(-1,2),(y-base)[complete].ravel(),rcond=None)[0]; v=F@coef; implied=(y-base)@v/(v@v); print('complete reset coefficients',coef,'diag imputation candidates');
for j in np.flatnonzero(~complete):
 k=np.flatnonzero(~np.isfinite(R[j])); print('lab,stratum',lab[j],st[j],'col',k,'implied shift',round(float(implied[j]-drift[j]),5),'current',xdiag[j,k], 'global standardized means',x[:,k].mean(0), 'zero delta',-xdiag[j,k]@beta[3+k], 'joint center delta',(x[:,k].mean(0)-xdiag[j,k])@beta[3+k])
