import runpy,pickle,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c01_batch_04_solver.py'); globals().update({k:v for k,v in m.items() if k.startswith('_') and not k.startswith('__')})
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=b['dataset']; tr,te,g=d['train'],d['test'],b['campaign_diagnostics']; cal=d['calibration_load_cells']; load,raw=cal['true_channel_loads'],cal['bridge_raw']; bridge,keep=_fit_bridge(raw,load); err=np.linalg.norm(_bridge_forward(load,bridge)-raw,axis=1); print('cal retained',keep.sum(),'abs error percentiles',np.percentile(err[keep],[10,50,90,100]),'coeffs',bridge)
dt,t=d['past_dt'],d['eval_times']; xtr,xte=_features(tr,te); a,h,f=_calibrated_inputs(tr,xtr,bridge); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,th=_fit(a,h,f,dt,t,y,r); print('train intrinsic:',tau,th); cc,rr=_design(tau,a,h,f,dt,t); print('train residual',np.percentile(abs(cc@th-y),[25,50,75,90]),np.percentile(abs(rr@th-r),[25,50,75,90]));
for label,xg in [('train_diag',_features(tr,g)[1]),('pooled_diag',_features(dict({k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}),g)[1])]:
 p,H,F=_calibrated_inputs(g,xg,bridge); Y=np.column_stack((g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs'])); drift=F@th[2:]; R=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]; age=np.cumsum(dt[::-1])[::-1]-dt; Z=p@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); base=(H[:,None]*th[:2])@R.T
 def calc(A,y=Y):
  mat=A.reshape(Y.size,-1); y=y.ravel(); init=np.linalg.lstsq(mat,y,rcond=None)[0]; return least_squares(lambda v: mat@v-y,init,loss='cauchy',f_scale=.006,max_nfev=300).x
 def report(name,pred,par):
  e=pred-Y; print(label,name,np.round(par,5),'MAD',np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])), np.median(e[:,-1])],5),'bytime',np.round(np.median(e,axis=0),4))
 for nm,predictors in [('projection',drift[:,None]),('hold+drift',np.c_[H,drift]),('hold+memory+drift',np.c_[H,Z,drift]),('hold+memory+features',np.c_[H,Z,F])]:
  A=np.einsum('ni,tj->ntij',predictors,R).reshape(len(H),len(t)+1,-1); v=calc(A,Y-base); report(nm,base+A@v,v.reshape(-1,2))
 A=drift[:,None,None]*R[None,:,:]; uncon=calc(A,Y-base); report('archived projection',base+A@np.array([.188729,2.002465]),[.188729,2.002465]);
 D=np.outer(drift,np.r_[t,1]); report('reset+linear',base+D,[]); report('ordinary',base-(Z*th[:2])@R.T+D,[]);
 for nm,pred in [('reset',base),('ordinary',base-(Z*th[:2])@R.T)]:
  f0=np.r_[t,1.]; mat=drift[:,None]
  ramp=np.array([calc(mat*np.eye(len(t)+1)[None,j,:] if False else np.broadcast_to(mat,(len(H),1))[:,None,:] * np.eye(len(t)+1)[None,:,j,None],Y-pred)[0] for j in range(len(t)+1)]); print(label,nm,'free drift ramp',np.round(ramp,4))
