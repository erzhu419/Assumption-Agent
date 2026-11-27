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
for name,g,X,P,H in [('train',tr,x,p,h),('diag',c,xc,pc,hc)]:
 Y=np.column_stack((g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']));a,r=m._design(tau,X,P,H,dt,t);fit=np.column_stack((a@beta,r@beta));e=Y-fit
 print(name,'baseline residual quantiles by time',np.quantile(e,[.1,.5,.9],axis=0)); print('residual singular values',np.linalg.svd(e,compute_uv=False));print('load range',np.min(H),np.max(H),np.std(P,axis=0))
 def amplitudes(ta):
  F=np.vstack((np.column_stack((-np.expm1(-t[:,None]/ta),t)),np.r_[1/ta,1.]));co=np.linalg.lstsq(F,Y.T,rcond=None)[0].T;return co,co@F.T-Y
 for tt in [tau,np.sort(np.exp(least_squares(lambda z:amplitudes(np.exp(z))[1].ravel(),np.log(tau),bounds=(np.log(.052),np.log(24.9)),loss='soft_l1',f_scale=.015).x))]:
  co,re=amplitudes(tt); print('free amp taus',tt,'medabs error',np.median(abs(re)),'per-time RMSE',np.sqrt(np.mean(re**2,axis=0)))
  C=np.column_stack((np.ones(len(H)),H,P,X)); bb=np.linalg.lstsq(C,co,rcond=None)[0]; print('amp linear coeff',bb.T,'amp residual RMS',np.sqrt(np.mean((co-C@bb)**2,axis=0)))
