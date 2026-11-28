import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/alpha_m100_c02_batch_05_solver.py'; s=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=a['dataset']; tr,te,g=d['train'],d['test'],a['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; samp=m._bridge_samples(cal); co=m._fit_bridge(*samp)
L,R,c,U,V=samp
print('bridge coefficients',co.round(6)); print('bridge RMSE',np.mean((m._forward_bridge(L,np.roll(L,2,axis=1),co)-R)**2)**.5,np.mean((co[c]+np.r_[1,co[4:7]][c]*(m._bridge_basis(U,np.zeros_like(U))@co[7:])-V)**2)**.5)
X=m._features(tr,te); P,H=m._loads(co,tr); Y=tr['strain_curves_obs']-tr['strain_0'][:,None]; I=tr['instant_rate_obs']; tau,th=m._fit(P,H,X[:len(H)],dt,t,Y,I); print('tau, theta',tau,th); A,B,*_=m._design(np.log(tau),P,H,X[:len(H)],dt,t); print('train curve, rate abs residual quantiles',*[np.quantile(np.abs(q),[.5,.9,.99,1]) for q in (A@th-Y,B@th-I)])
p,h=m._loads(co,g); y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; x=g['coupon_features_raw'].copy(); labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]);
for lab in np.unique(labs):
 rows=g['lab_id']==lab; panel=refs[labs==lab]; x[rows]=(x[rows]-np.median(panel.mean(0)))/np.median(panel.std(0));
mask=~np.isnan(x).any(1); x=np.nan_to_num(x); z=0*p[:,:2]
for dur,stress in zip(dt,p.T):
 f=np.exp(-dur/tau);z=f*z+(1-f)*stress[:,None]
drift=np.c_[np.ones(len(h)),x]@th[2:]; basis=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]])
v=np.c_[y,r]; amps=v@np.linalg.pinv(basis.T); print('diagnostic within-coupon exp RMSE curve/rate',np.sqrt(np.mean((amps@basis.T-v)**2,axis=0)).round(5)); print('baseline curve/rate',np.sqrt(np.mean(((np.c_[(h[:,None]-z)*th[:2],drift]@basis.T-v)**2)[mask],axis=0)).round(5));
for name,W in [('hold+drift',np.c_[np.ones(len(h)),h,drift]),('hold+drift+memory',np.c_[np.ones(len(h)),h,drift,z]),('hold+drift+history',np.c_[np.ones(len(h)),h,drift,p]),('hold+features+memory',np.c_[np.ones(len(h)),h,x,z])]:
 C=np.linalg.lstsq(W[mask],amps[mask],rcond=None)[0]; q=[]
 for lab in np.unique(g['lab_id']):
  f=mask&(g['lab_id']!=lab); e=mask&~f; b=np.linalg.lstsq(W[f],amps[f],rcond=None)[0]; err=W[e]@b@basis.T-v[e]; q.append([np.mean(err[:,:-1]**2)**.5,np.mean(err[:,-1]**2)**.5]);
 print(name,'coef',C.round(4).tolist(),'LOLO',np.array(q).round(4).tolist())
print('coupon delta amplitudes rows: hold, drift, z0, z1, amps + complete-case flag');print(np.c_[h,drift,z,amps,mask].round(4))
