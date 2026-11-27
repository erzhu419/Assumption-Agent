import pickle,numpy as np
from m98_core import _stress_calibration,_standardize,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=140)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; train,test,diag=d['train'],d['test'],i['campaign_diagnostics']; c=d['calibration_load_cells']; t,dt=d['eval_times'],d['past_dt']; stress=_stress_calibration(c); print('cal inverse clean max error',np.max(abs(stress(c['bridge_raw'][np.max(abs(c['bridge_raw']),1)<5])-c['true_channel_loads'][np.max(abs(c['bridge_raw']),1)<5].sum(1))))
x,xt=_standardize(train,test); matrices=_design(train,x,stress,dt,t); y=train['strain_curves_obs']-train['strain_0'][:,None]; r=train['instant_rate_obs']; taus,theta=_fit(matrices,y,r); print('taus theta',taus,theta)
def metrics(v):return np.round([np.median(abs(v)),np.sqrt(np.mean(v*v)),np.max(abs(v))],6)
s0,r0=matrices(taus); print('benchtop fit: medAE, RMSE, maxAE',metrics(s0@theta-y),metrics(r0@theta-r))
for k in ['train','test']:
 s=d[k]; print(k,'telemetry',np.quantile(s['aux_chamber_telemetry'],[0,.25,.5,.75,1],axis=0),s['aux_chamber_telemetry'][:3]); print(k,'physical loads',np.quantile(stress(s['past_channel_raw']),[0,.25,.5,.75,1]),np.quantile(stress(s['hold_channel_raw']),[0,.25,.5,.75,1]))
_,xd=_standardize(train,diag); xd2,_=_standardize(diag,diag); yd=diag['strain_curves_obs']-diag['strain_0'][:,None]; rd=diag['instant_rate_obs']; past=stress(diag['past_channel_raw']); hold=stress(diag['hold_channel_raw']); print('diag x alternatives RMS difference',np.sqrt(np.mean((xd-xd2)**2)))
for xdia in [xd,xd2]:
 drift=theta[2]+xdia@theta[3:]; print('drift ranges', np.min(drift),np.max(drift));
 for memory in ['normal','reset','reverse']:
  h=past if memory=='normal' else past[:,::-1] if memory=='reverse' else 0*past; tail=np.cumsum(dt[::-1])[::-1]-dt; state=h@(-np.expm1(-dt[:,None]/taus)*np.exp(-tail[:,None]/taus)); gap=hold[:,None]-state; branch=(gap*theta[:2])@(-np.expm1(-t[:,None]/taus)).T; init=gap@(theta[:2]/taus);
  for name,time in [('linear',t),('saturate',-np.expm1(-t)),('log',np.log1p(t)),('rational',t/(1+t))]:
   err=branch+drift[:,None]*time-yd; print(memory,name,metrics(err),metrics(init+drift-rd));
 print('hold|rate|train-nominal drift|observed early|observed late',np.c_[hold,rd,drift,yd[:,0],yd[:,-1]])
