import numpy as np,pickle,json
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from scipy.special import erf
q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=q['dataset'];c=d['calibration_load_cells'];u=c['true_channel_loads'];r=c['bridge_raw'];s=c['single_channel_sweeps']; np.set_printoptions(precision=6,suppress=True,linewidth=150)
for j in range(4):
 m=s['lab_id']==j; x=s['true_stress'][m]; y=s['cmd_stress'][m]; a=x[abs(y)<3]; y=y[abs(y)<3]; p=least_squares(lambda p:p[0]+p[1]*a+p[2]*np.sin(p[3]*a)-y,[0,1,.25,2.5],loss='soft_l1',f_scale=.0003).x
 v=r[:,j]-p[0]-p[1]*u[:,j]-p[2]*np.sin(p[3]*u[:,j]); keep=abs(r[:,j])<4
 print('channel',j,'axis fit',p)
 print('axis residual',v[:16].reshape(4,4))
 for name,A in [('poly2',np.column_stack([np.prod(u[:,ii],axis=1) for k in [1,2] for ii in combinations_with_replacement(range(4),k)])),('poly3',np.column_stack([np.prod(u[:,ii],axis=1) for k in [1,2,3] for ii in combinations_with_replacement(range(4),k)])),('structured',np.c_[u,u**2,np.sin(2.5*u),np.tanh(u),np.tanh(2.5*u),np.cos(2.5*u)-1,*[u[:,k]*u[:,l] for k,l in combinations_with_replacement(range(4),2) if k!=l]])]:
  b=np.linalg.lstsq(A[keep],v[keep],rcond=None)[0]; err=A@b-v; print(name,np.quantile(abs(err[keep]),[.5,.9,1]),b.round(4))
for k in ['train','test']:
 v=d[k]['aux_chamber_telemetry']; print(k,'aux mean std',v.mean(0),v.std(0));print('aux first',v[:3])
print('diag',q.get('campaign_diagnostics') is not None)
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],'mean',a.mean(0).round(3),'max',a.max(0).round(3),'tail',a[-3:,[1,6]].round(3))
