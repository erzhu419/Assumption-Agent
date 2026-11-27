import sys,pickle,json,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from m99_base import _fit,_features,_design
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
raw=item['dataset']; dg=item['campaign_diagnostics']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
d=calibrate_upstream(raw,pay); tr,te=d['train'],d['test']; dg=calibrate_upstream(dict(raw,test=dg),pay)['test']
t,dt=d['eval_times'],d['past_dt']; x,xt=_features(tr,te)
# Use the specified normalization on train+test; map diagnostics against the same reference population.
parts=[tr,te]; rx=np.vstack([p['coupon_features_raw'] for p in parts]); rr=np.vstack([p['ref_coupons'] for p in parts]); labs=np.r_[tr['lab_id'],te['lab_id']]; xd=dg['coupon_features_raw'].copy()
for l in np.unique(labs):
 r=rr[labs==l]; xd[dg['lab_id']==l]=(xd[dg['lab_id']==l]-np.median(r.mean(0)))/np.median(r.std(0))
for j in range(4):
 m=np.isnan(rx[:,j]); xd[np.isnan(xd[:,j]),j]=np.vstack([x,xt])[m,j][0]
def loads(p):return p['past_channel_raw'].sum(-1),p['hold_channel_raw'].sum(-1)
p,h=loads(tr); pd,hd=loads(dg); Y0=tr['strain_curves_obs']-tr['strain_0'][:,None]; R0=tr['instant_rate_obs']; Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; tau,beta=_fit(x,p,h,dt,t,Y0,R0)
if __name__=='__main__':
 print('tau,beta',tau,beta,'hold',hd.round(3)); A,B=_design(tau,x,p,h,dt,t); print('tr residual quantiles',np.quantile(abs(A@beta-Y0),[.5,.9,.99,1]),np.quantile(abs(B@beta-R0),[.5,.9,.99,1])); A,B=_design(tau,xd,pd,hd,dt,t); print('benchmark residual',np.sqrt(np.mean((A@beta-Y)**2)),np.sqrt(np.mean((B@beta-R)**2)))
 def modal(ta,extra=0):
  K=np.c_[-np.expm1(-t[:,None]/ta),t]; k=np.r_[1/ta,1];
  if extra: K=np.c_[K,t*t]; k=np.r_[k,0]
  M=np.r_[K,k[None,:]]; Z=np.c_[Y,R]; C=np.linalg.lstsq(M,Z.T,rcond=None)[0].T; return C,C@M.T-Z
 for ta in [tau,tau/2,tau*2,np.sort(np.exp(least_squares(lambda u:modal(np.exp(u))[1].ravel(),np.log(tau),bounds=(np.log(.025),np.log(60))).x))]:
  C,E=modal(ta);print('modal',ta,'error quantiles',np.quantile(abs(E),[.5,.9,.99,1]),'coeff ranges',np.quantile(C,[0,.5,1],axis=0));
  drift=beta[2]+xd@beta[3:]; z=hd[:,None]-A[:,0,:2]/(-np.expm1(-t[0]/tau)); G=np.c_[np.ones(len(C)),hd,z,drift,xd,pd]; print('corr',np.corrcoef(np.c_[C,G[:,1:]].T)[:3,3:].round(2)); print('resid shape',E[:3].round(4));
 print('telemetry means',[(q['completed_campaign_index'],np.mean(q['facility_telemetry_trace'],axis=0).round(2)) for q in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))]); print('diag sample',np.c_[Y[:5],R[:5]].round(4))
