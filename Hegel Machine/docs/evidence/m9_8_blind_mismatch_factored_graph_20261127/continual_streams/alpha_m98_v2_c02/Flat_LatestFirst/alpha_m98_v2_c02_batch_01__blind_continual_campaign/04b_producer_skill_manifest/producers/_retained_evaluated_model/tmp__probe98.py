import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m98_v2_c02_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; ck=p['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; cal=d['calibration_load_cells']; stress=m._stress_calibration(cal)
xt,xe=m._standardize(tr,te); allsite={k:np.concatenate([tr[k],te[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; _,xc=m._standardize(allsite,ck)
get=lambda g:(g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs'])
z,w=m._fit(m._design(tr,xt,stress,dt,t),*get(tr)); print('dt/t',dt,t,'fit',z,w)
print('bridge q',np.quantile(stress(cal['bridge_raw'])-cal['true_channel_loads'].sum(1),[0,.1,.5,.9,1])); print('diagnostic fields',list(ck),'missing',np.isnan(ck['coupon_features_raw']).sum(0))
for g,x in [(tr,xt),(ck,xc)]:
 y,r=get(g); a,b=m._design(g,x,stress,dt,t)(z); e=y-a@w; print('baseline per time bias/rmse',np.median(e,axis=0),np.mean(e**2,axis=0)**.5,'rate',np.quantile(r-b@w,[0,.1,.5,.9,1]))
y,r=get(ck); target=np.c_[y,r]; a,b=m._design(ck,xc,stress,dt,t)(z)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def profile(taus):
 q=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]])
 beta=np.linalg.lstsq(q,target.T,rcond=None)[0].T
 return beta, beta@q.T-target
for zz in [z, np.sort(np.exp(least_squares(lambda u:profile(np.sort(np.exp(u)))[1].ravel(),np.log(z),max_nfev=100).x))]:
 beta,e=profile(zz); gap=b[:,:2]*z; print('profile',zz,'rmse curve/rate',np.mean(e[:,:-1]**2)**.5,np.mean(e[:,-1]**2)**.5); print('columns hold / inferred amp1,amp2,drift / amp ratios / drift delta / residual RMS'); print(np.c_[stress(ck['hold_channel_raw']),beta,beta[:,:2]/(gap*w[:2]),beta[:,2]-b[:,2:]@w[2:],np.mean(e**2,axis=1)**.5])
