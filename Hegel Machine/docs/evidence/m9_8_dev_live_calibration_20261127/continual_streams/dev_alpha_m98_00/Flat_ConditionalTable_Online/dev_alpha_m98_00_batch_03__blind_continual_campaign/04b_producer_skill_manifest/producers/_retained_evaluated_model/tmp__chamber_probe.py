import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_alpha_m98_00_s00_solver.py'
s=importlib.util.spec_from_file_location('base',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; tc,ec=m._prepare(d)
tau,theta=m._fit(dt,t,tc,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,theta)
refs=np.concatenate([tr['ref_coupons'],te['ref_coupons']]); labs=np.concatenate([tr['lab_id'],te['lab_id']]); x=q['coupon_features_raw'].copy(); print('diag labs and missing',np.unique(q['lab_id'],return_counts=True),np.isnan(x).sum(0)); print('diag stratum',q['stratum']);
for lab in np.unique(q['lab_id']):
 r=refs[labs==lab]; x[q['lab_id']==lab]=(x[q['lab_id']==lab]-np.median(np.nanmean(r,0)))/np.median(np.nanstd(r,0))
lb=np.r_[tr['lab_id'],te['lab_id']]; st=np.r_[tr['stratum'],te['stratum']]; raw=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); xf=np.vstack((tc[2][:,1:],ec[2][:,1:])); _,j,n=np.unique(np.c_[lb,st],axis=0,return_inverse=True,return_counts=True); w=(~np.isnan(raw))/n[j,None]; x=np.where(np.isnan(x),(w*xf).sum(0)/w.sum(0),x); x=np.c_[np.ones(len(x)),x]
cal=m._stress_calibration(d['calibration_load_cells']); c=d['calibration_load_cells']; print('affine cal residual abs',np.quantile(np.abs(c['bridge_raw']@cal[:-1]+cal[-1]-c['true_channel_loads'].sum(1)),[.5,.9,1])); past=q['past_channel_raw']@cal[:-1]+cal[-1]; hold=q['hold_channel_raw']@cal[:-1]+cal[-1]; cp=(past,hold,x); A,B=m._design(tau,dt,t,cp); y=q['strain_curves_obs']-q['strain_0'][:,None]; v=q['instant_rate_obs']; g=-np.expm1(-t[:,None]/tau); drift=x@theta[2:]; z=hold[:,None]-B[:,:2]*tau

def report(name,yh,vh):
 for err in (yh-y,vh-v):
  print(name,err.shape,np.round(np.quantile(np.abs(err),[.5,.9,1]),6), 'rmse',np.sqrt(np.mean(err**2)))
report('benchtop',A@theta,B@theta)
for name,f in [('reset',[0,0]),('archive01',[.52,.43])]:
 f=np.array(f); amp=hold[:,None]*theta[:2]+drift[:,None]*f*tau; r=drift*(1-f.sum()); report(name,amp@g.T+r[:,None]*t,amp@(1/tau)+r)
# Recover each coupon's two amplitudes and steady slope, including the measured derivative.
from scipy.optimize import least_squares
D=np.c_[g,t]; Dr=np.r_[1/tau,1]; Q=np.vstack((D,Dr)); coef=np.array([least_squares(lambda a:Q@a-b,np.linalg.lstsq(Q,b,rcond=None)[0],loss='soft_l1',f_scale=.0001).x for b in np.c_[y,v]])
print('recovered [hold, z0,z1,drift, rate residual vs reset, adjusted A0,A1,r]', np.round(np.c_[hold,z,drift,v-hold@np.ones((1,)) if False else v-hold*np.sum(theta[:2]/tau)-drift,coef[:,:2]-hold[:,None]*theta[:2],coef[:,2]],6)); print('test telemetry quantiles',np.quantile(te['aux_chamber_telemetry'],[0,.5,1],axis=0))
