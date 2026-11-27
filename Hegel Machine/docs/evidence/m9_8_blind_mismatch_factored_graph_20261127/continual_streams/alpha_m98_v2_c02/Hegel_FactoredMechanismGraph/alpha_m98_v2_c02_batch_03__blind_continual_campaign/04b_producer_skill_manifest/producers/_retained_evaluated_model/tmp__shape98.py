import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c02_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; te=d['test']; c=i['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; st=m._stress_calibration(d['calibration_load_cells']); xtr,xte=m._standardize(tr,te); _,xc=m._standardize(tr,c)
tau,th=m._fit(m._design(tr,xtr,st,dt,t),tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau,theta',tau,th)
S,R=m._design(c,xc,st,dt,t)(tau); y=c['strain_curves_obs']-c['strain_0'][:,None]; r=c['instant_rate_obs']; n=y.size
A=np.vstack((S.reshape(-1,7),R)); Y=np.r_[y.ravel(),r]; base=A@th-Y
print('base median/RMSE',[(np.median(abs(v)),np.std(v),np.sqrt(np.mean(v*v))) for v in (base[:n],base[n:])]);
for ix in ([0,1],list(range(7))):
 B=A[:,ix]; target=B@th[ix]-base; sol=least_squares(lambda w:(B@w-target)/np.r_[np.full(n,.003),np.full(len(r),.006)],th[ix],loss='soft_l1'); print('coeff',ix,sol.x,'gains',sol.x/th[ix]); res=B@sol.x-target; print('errors',[(np.median(abs(v)),np.sqrt(np.mean(v*v))) for v in (res[:n],res[n:])]); print('per lab',[(j,np.median(abs(res[:n].reshape(y.shape)[c['lab_id']==j])),np.median(abs(res[n:][c['lab_id']==j]))) for j in np.unique(c['lab_id'])]);
from scipy.optimize import least_squares
b=S[:,:,:2]@th[:2]; br=R[:,:2]@th[:2]; yr=y-b; rr=r-br
print('telemetry fields',[(key,type(d['test'][key]),getattr(d['test'][key],'shape',None)) for key in d['test'] if 'telemetry' in key]); print('references',tr['ref_coupons'].shape,te['ref_coupons'].shape,c['ref_coupons'].shape)
print('drift by coupon: lab, inferred onset, reference-predicted, last strain per unit inferred onset'); print(np.round(np.column_stack((c['lab_id'],rr,R[:,2:]@th[2:],yr[:,-1]/rr)),5));
for lab in np.unique(c['lab_id']):
 for name,site in [('tr',tr),('te',te),('diag',c)]:
  z=site['ref_coupons'][site['lab_id']==lab]; print(lab,name,round(np.nanmedian(np.nanmean(z,axis=0)),4),round(np.nanmedian(np.nanstd(z,axis=0)),4))
# Profile the shared time response with nuisance coupon amplitudes (including onset data).
fit=least_squares(lambda p: np.r_[((p[12:,None]*p[:12]-yr)/.004).ravel(),(p[12:]-rr)/.006],np.r_[.5*t,rr],loss='cauchy',f_scale=2)
print('times',t); print('time shape',fit.x[:12]); print('nuisance amplitude',fit.x[12:]);print('curve/rate residual median',np.median(abs(fit.fun[:n]))*.004,np.median(abs(fit.fun[n:]))*.006)
