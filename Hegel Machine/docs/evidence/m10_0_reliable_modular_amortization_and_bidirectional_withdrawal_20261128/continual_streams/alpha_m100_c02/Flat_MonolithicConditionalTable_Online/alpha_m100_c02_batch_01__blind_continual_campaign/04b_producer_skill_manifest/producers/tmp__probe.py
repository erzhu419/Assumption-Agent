import pickle,numpy as np
from chamber_base import _features,_loads,_fit,_design,solve_batch
np.set_printoptions(precision=5,suppress=True,linewidth=140)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; ca=d['calibration_load_cells']; dt=d['past_dt']; t=d['eval_times']; s=solve_batch(d); tau=s['estimated_taus']; th=s['hereditary_weights']; print('material',tau,th)
def summary(r): return np.round([np.sqrt(np.mean(r*r)), np.median(abs(r)),np.quantile(abs(r),.9),np.max(abs(r))],7)
xtr=_features(tr,te)[:len(tr['lab_id'])]; xq=_features(dict((k,np.concatenate([tr[k],te[k]])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):];
for name,u,x in [('train',tr,xtr),('diag',q,xq)]:
 p,h=_loads(ca,u); C,R,*_=_design(np.log(tau),p,h,x,dt,t); Y=u['strain_curves_obs']-u['strain_0'][:,None]; r=u['instant_rate_obs']; print(name,'curve/rate',summary(Y-C@th),summary(r-R@th),'ranges',np.min(p),np.max(p),np.min(h),np.max(h));
 T=np.column_stack([-np.expm1(-t[:,None]/tau),t]); A=np.vstack((T,np.r_[1/tau,1]*.4)); coef=np.linalg.lstsq(A,np.column_stack((Y,r*.4)).T,rcond=None)[0].T; e=np.column_stack((Y,r*.4))-coef@A.T; print('time-basis residual',summary(e[:,:-1]),summary(e[:,-1]));
 if name=='diag':
  exp=np.column_stack((R[:,:2]*tau*th[:2],x@th[2:])); print('rows [lab,stratum, predicted coefficients, observed coefficients, delta rate, row residual RMS]'); print(np.column_stack([u['lab_id'],u['stratum'],exp,coef,r-R@th,(e**2).mean(1)**.5])); print('curve delta median per time',np.median(Y-C@th,axis=0)); print('residual singular values',np.linalg.svd(Y-C@th,compute_uv=False)); print('diag fit as benchtop',_fit(p,h,x,dt,t,Y,r))
raw=ca['bridge_raw']; true=ca['true_channel_loads']; aa=np.c_[np.ones(len(true)),true]; print('bridge residual',summary(raw-aa@np.linalg.lstsq(aa,raw,rcond=None)[0])); print('single sweeps',[(k,np.shape(v),str(v)[:140]) for k,v in ca['single_channel_sweeps'].items()]); print('lab-stratum counts',[(n,np.unique(np.c_[u['lab_id'],u['stratum']],axis=0,return_counts=True)[1]) for n,u in [('tr',tr),('te',te),('q',q)]])
