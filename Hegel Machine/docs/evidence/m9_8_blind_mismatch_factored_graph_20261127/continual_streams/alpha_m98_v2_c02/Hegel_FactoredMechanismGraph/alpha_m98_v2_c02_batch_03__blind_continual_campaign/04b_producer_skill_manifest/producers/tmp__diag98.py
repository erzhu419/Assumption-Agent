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
G=np.column_stack((-np.expm1(-t[:,None]/tau),t)); abc=np.linalg.lstsq(np.vstack((G, np.r_[1/tau,1])),np.vstack((y.T,r)),rcond=None)[0].T; print('per coupon [hold,base gap0,gap1, branch0,branch1, drift, baseline drift]'); print(np.round(np.column_stack((st(c['hold_channel_raw']),R[:,:2]*tau,abc,R[:,2:]@th[2:])),5)); print('linear basis residual',np.median(abs(abc@G.T-y)),np.median(abs(abc@np.r_[1/tau,1]-r)))
