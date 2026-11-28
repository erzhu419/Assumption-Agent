import pickle,json
import numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=170)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; T=c['true_channel_loads']; R=c['bridge_raw']; q=c['single_channel_sweeps']; ok=np.max(abs(R),axis=1)<5; axis=np.count_nonzero(T,axis=1)==1
for j in range(4):
 p=j^2; sw=(q['lab_id']==j)&(abs(q['cmd_stress'])<5); ax=ok&axis&(T[:,j]!=0); x=np.r_[q['true_stress'][sw],T[ax,j]]; y=np.r_[q['cmd_stress'][sw], R[ax,j]]
 f=lambda a,u:a[0]+a[1]*u+a[2]*np.sin(a[3]*u)
 fit=least_squares(lambda a:f(a,x)-y,[0,1,.2,2.5]).x
 print('self',j,fit,'rmse',np.std(f(fit,x)-y))
 a=T[ok,j]; b=T[ok,p]; z=R[ok,j]-f(fit,a); u,v=T[ok,j],T[ok,p]
 print('pair axis',np.c_[b[(a==0)&(b!=0)],z[(a==0)&(b!=0)]])
 for opt in ['poly2','tanh','ration','cos','trig','mixed','full']:
  feat=[b];
  if opt=='poly2':feat += [b*b,u*v]
  if opt=='tanh':feat += [np.tanh(b)**2,u*v]
  if opt=='ration':feat += [b*b/(1+b*b),u*v]
  if opt=='cos':feat += [1-np.cos(1.5*b),u*v]
  if opt=='trig':feat += [np.sin(2.5*b),1-np.cos(2.5*b),u*v]
  if opt=='mixed':feat += [np.sin(2.5*b),b*b/(1+b*b),u*v,np.sin(2.5*u)*b,u*np.sin(2.5*b),np.sin(2.5*u)*np.sin(2.5*b)]
  if opt=='full':
   cols=T[ok].T; feat=list(cols)+[w*w for w in cols]+[np.sin(2.5*w) for w in cols]+[cols[k]*cols[l] for k in range(4) for l in range(k)]
  F=np.array(feat).T; err=z-F@np.linalg.lstsq(F,z,rcond=None)[0]; print(opt, np.sqrt(np.mean(err**2)), np.max(abs(err)))
 print('mixed example: own paired residual',np.c_[a,b,z][~axis[ok]][:8])
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('telemetry campaigns',[(e['completed_campaign_short_name'],type(e['facility_telemetry_trace']).__name__,len(e['facility_telemetry_trace'])) for e in log]); print('last trace fields', list(log[-1]['facility_telemetry_trace'])[:12] if isinstance(log[-1]['facility_telemetry_trace'],dict) else list(log[-1]['facility_telemetry_trace'][0])[:12])
