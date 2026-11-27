import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; q=item['campaign_diagnostics']; train=d['train']; dt,t=d['past_dt'],d['eval_times']; cp,deploy=m._prepare(d)
obs=train['strain_curves_obs']-train['strain_0'][:,None]; v=train['instant_rate_obs']; out=m.solve_batch(d); tau,b=out['estimated_taus'],out['hereditary_weights']; assert all(np.isfinite(a).all() for a in out.values()); assert .05 < tau[0] < tau[1] < 25

def rms(e): return np.sqrt(np.mean(e**2))
for lab in np.unique(train['lab_id']):
    val=train['lab_id']==lab; fit=~val
    tt,bb=m._fit(dt,t,tuple(a[fit] for a in cp),obs[fit],v[fit]); C,R=m._design(tt,dt,t,tuple(a[val] for a in cp)); e=C@bb-obs[val]; er=R@bb-v[val]; print('train held out lab',lab,'strain and rate RMSE',rms(e),rms(er),'median abs strain',np.median(np.abs(e)),'taus',tt)
keys=('lab_id','stratum','ref_coupons','coupon_features_raw'); cut=len(q['lab_id'])//2
x=m._aligned_features({k:q[k][:cut] for k in keys},{k:q[k][cut:] for k in keys}); cal=m._stress_calibration(d['calibration_load_cells']); diag=(q['past_channel_raw']@cal[:-1]+cal[-1],q['hold_channel_raw']@cal[:-1]+cal[-1],x)
obs=q['strain_curves_obs']-q['strain_0'][:,None]; v=q['instant_rate_obs']; P,V=m._chamber(tau,b,dt,t,diag); print('final chamber diagnostics RMSE',rms(P-obs),rms(V-v)); C,R=m._design(tau,dt,t,diag); base=C@b; dr=x@b[2:]; g=-np.expm1(-t[:,None]/tau)*tau-t[:,None]; D=dr[:,None,None]*g
for lab in np.unique(q['lab_id']):
    val=q['lab_id']==lab; fit=~val
    par=least_squares(lambda a:(base[fit]+D[fit]@a-obs[fit]).ravel(),[.6,.6],loss='soft_l1',f_scale=.02).x; e=base[val]+D[val]@par-obs[val]; print('chamber held out lab',lab,'strain and rate RMSE',rms(e),rms((R@b-v)[val]),'fractions',par)
h=1e-6; p0,r=m._chamber(tau,b,dt,np.array([0.,h]),deploy); np.testing.assert_allclose(p0[:,0],0,atol=1e-14); np.testing.assert_allclose(p0[:,1]/h,r,atol=1e-5,rtol=1e-5); print('boundary derivative max error',np.max(np.abs(p0[:,1]/h-r)))
