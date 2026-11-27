import pickle, numpy as np
from scipy.optimize import least_squares
import prior as p
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; g=it['campaign_diagnostics']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; s=p._stress_map(d['calibration_load_cells']); tau,th=p._fit(tr,p._features(tr,te)[0],s,dt,t); h=s[0]+g['hold_channel_raw']@s[1:]; Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; labs=g['lab_id']; miss=np.isnan(g['coupon_features_raw']); x=p._features(g,g)[0]; r=th[2]+x@th[3:];
def ker(u):
 u=np.atleast_1d(u);return np.vstack([-np.expm1(-t[:,None]/u),1/u])
E=Y-(h[:,None]*th[:2])@ker(tau).T; _,_,vh=np.linalg.svd(E-E.mean(0),full_matrices=False); v=vh[0]; targets=np.vstack([E.mean(0),v]); line=np.r_[t,1.][:,None]
for name, fn, init in [('one',lambda u:ker(u),[2.4]),('one+line',lambda u:np.c_[ker(u),line],[2.4]),('one+fast',lambda u:np.c_[ker(u),ker(tau[:1])],[2.4]),('two',ker,[.55,2.4]),('two+line',lambda u:np.c_[ker(u),line],[.55,2.4]),('fixed log rate',lambda u:np.c_[np.log1p(t/u[0]).tolist()+[1/u[0]]], [2.4])]:
 def res(q, yy):
  a=fn(np.exp(q)); return (np.linalg.lstsq(a,yy.T,rcond=None)[0].T@a.T-yy).ravel()
 f=least_squares(lambda q:res(q,targets),np.log(init),bounds=(np.log(.04),np.log(30))); u=np.exp(f.x); a=fn(u); print(name, 'tau',u,'fit',np.sqrt(np.mean(f.fun**2)),'coeff',np.linalg.lstsq(a,targets.T,rcond=None)[0].T)
# Regress the chamber transient on the complete standardized coupons, then back out missing entries.
i=~miss.any(1); Z=np.c_[np.ones(len(r)),x]; b=np.linalg.lstsq(Z[i],(E@v)[i],rcond=None)[0]; print('scaled transient regression',b/b[0], 'intrinsic',th[3:]/th[2]); print('complete/missing projection errors',np.round(E@v-Z@b,5)); print('observed imputation in x',np.where(miss,x,np.nan)); print('backsolved missing by row',[(int(j),int(k),float(x[j,k]+(E@v-Z@b)[j]/b[k+1])) for j,k in zip(*np.where(miss))]);
for group in (tr,te):
 a=group['aux_chamber_telemetry']; print('aux summary mean std',a.mean(0),a.std(0),'ranges',np.min(a,axis=0),np.max(a,axis=0));
