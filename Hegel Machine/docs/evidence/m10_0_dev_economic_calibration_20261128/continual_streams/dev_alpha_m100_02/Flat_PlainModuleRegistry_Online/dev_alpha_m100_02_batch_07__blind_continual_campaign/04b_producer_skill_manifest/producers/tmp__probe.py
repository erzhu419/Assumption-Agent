import json,pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
import material_probe as m
item=pickle.load(open('public_batch.pkl','rb'))[0]; raw=item['dataset']; pay=json.load(open('_upstream_param_payload.json')); d=calibrate_upstream(raw,pay); q=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),pay)['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; x,xte=m._features(tr,te); tau,theta=m._material(tr,x,dt,t); print('material',tau,theta)
a=m._design(tau,tr['past_channel_raw'].sum(-1),tr['hold_channel_raw'].sum(-1),x,dt,t); y=np.r_[(tr['strain_curves_obs']-tr['strain_0'][:,None]).ravel(),tr['instant_rate_obs']]; print('train abs q',np.quantile(abs(a@theta-y),[.5,.9,.99,1]))
p=q['past_channel_raw'].sum(-1); h=q['hold_channel_raw'].sum(-1); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; b,br=m._basis(tau,t); amp=np.c_[y,r]@np.linalg.pinv(np.vstack((b,br))).T; err=np.c_[y,r]-amp@np.vstack((b,br)).T; print('modal fit',np.quantile(abs(err),[.5,.9,.99,1]),'amp ranges',np.min(amp,0),np.max(amp,0))
for name,xq in [('alone',m._features(q)[0]),('pooled',m._features(tr,te,q)[-1])]:
 f=xq@theta[2:]; st=np.zeros((len(h),2));
 for dur,s in zip(dt,p.T): st+=-np.expm1(-dur/tau)*(s[:,None]-st)
 for label,pred in [('bench',np.c_[(h[:,None]-st)*theta[:2],f]),('batch01',np.c_[h[:,None]*theta[:2]+f[:,None]*(tau*np.array([.45,.52])),.03*f])]:
  er=pred@np.vstack((b,br)).T-np.c_[y,r]; print(name,label,'curve/rate RMSE',np.sqrt(np.mean(er[:,:-1]**2)),np.sqrt(np.mean(er[:,-1]**2)))
 for i in range(3):
  z=amp[:,i]-(h*theta[i] if i<2 else 0); terms=np.c_[f,st,h]; w=np.linalg.lstsq(terms,z,rcond=None)[0]; print(name,'mode',i,'fit f/st/h',np.round(w,5),'amp resid',np.std(z-terms@w),'feature corr',np.round(np.corrcoef(np.c_[z,xq[:,1:]].T)[0,1:],3))
print('first modal rows',np.round(np.c_[h,amp][:9],4))
