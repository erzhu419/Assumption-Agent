import sys,pickle,json,importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p='/holdout_task/'
i=pickle.load(open(p+'public_batch.pkl','rb'))[0]; d=i['dataset']; dg=i['campaign_diagnostics']; pay=json.load(open(p+'_upstream_param_payload.json'))
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m99_00_batch_01_solver.py'); prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
c=calibrate_upstream(d,pay); cd=calibrate_upstream(dict(d,test=dg),pay)['test']; dt=np.array(d['past_dt']); t=np.array(d['eval_times']); br=np.r_[c['total_load_projection'][0],c['total_load_projection'][1]]
n=len(c['train']['lab_id']); nt=len(c['test']['lab_id']); x=prior._features(c['train'],c['test'],cd)
tr=prior._coupon(c['train'],x[:n],br); diag=prior._coupon(cd,x[n+nt:],br)
y=c['train']['strain_curves_obs']-c['train']['strain_0'][:,None]; r=c['train']['instant_rate_obs']; taus,theta=prior._fit(tr,dt,t,y,r)
Y=cd['strain_curves_obs']-cd['strain_0'][:,None]; R=cd['instant_rate_obs']; past,hold,dr=diag; drift=dr@theta[2:]; curve,rate=prior._design(taus,diag,dt,t)
if __name__=='__main__':
 print('dt',dt,'times',t,'bridge',br,'taus',taus,'theta',theta)
 for key,s in [('train',d['train']),('test',d['test']),('diag',dg)]:
  print(key,'lab',np.unique(s['lab_id'],return_counts=True),'missing',np.isnan(s['coupon_features_raw']).sum(0));
  if 'aux_chamber_telemetry' in s:
   a=s['aux_chamber_telemetry'];print('aux mean/std',a.mean(0),a.std(0),a[:3])
 print('cal fields',d['calibration_load_cells'].keys()); a=d['calibration_load_cells']; raw=a['bridge_raw']; L=a['true_channel_loads'].sum(1); b=prior._bridge(a); err=raw@b[:-1]+b[-1]-L;print('cal residual',np.quantile(abs(err),[0,.5,.9,.99,1]))
 for name,P,Q in [('base',curve@theta,rate@theta),('reset log',*prior._chamber(taus,theta,hold,drift,t)),('reset linear',hold[:,None]*(-np.expm1(-t[:,None]/taus)@theta[:2])+drift[:,None]*t, hold*(theta[:2]@(1/taus))+drift)]:
  print(name,'rmse by time',np.sqrt(np.mean((Y-P)**2,axis=0)).round(5),'rate',np.sqrt(np.mean((R-Q)**2))); print('signed median',np.median(Y-P,axis=0).round(5))
 np.set_printoptions(precision=4,suppress=True,linewidth=160); print('diag hold, drift, gap0, gap1, rate',np.c_[hold,drift,rate[:,:2]*taus,R]);print('diag Y first six',Y[:6]);
