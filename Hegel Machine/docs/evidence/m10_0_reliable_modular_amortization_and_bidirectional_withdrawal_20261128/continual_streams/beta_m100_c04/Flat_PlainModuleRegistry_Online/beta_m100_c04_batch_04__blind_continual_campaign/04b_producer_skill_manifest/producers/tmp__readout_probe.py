from wall_probe import *
from _verified_upstream import _fit, _basis
import json
s,w,y=load(); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['study']; sw=raw['sensor_calibration_sweeps']; x=sw['ref_rosette']; z=sw['raw_rosette']; p=json.load(open('/holdout_task/_upstream_param_payload.json')); np.set_printoptions(precision=5,suppress=True)
c=_fit(x,z); print('coef',c); print('sweep fit error percentiles',np.percentile(np.linalg.norm(_basis(x)@c-z,axis=1),[0,25,50,75,90,95,100])); print('single bank fields',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in sw['single_bank_sweeps'].items()]); print('vel endpoints',[raw['vel_'+q][[0,-1]] for q in 'xy']); print('shifts',s['mu_shifts']); print('ij',s['sensor_ij'].tolist()); print('bank',s['row_bank_id'][:32]);
r=np.vstack([d['y_pulse_raw'] for d in item['campaign_diagnostics']]).T; a=((r.reshape(-1,4,3)-np.array(p['offsets'])[None,:,None])/np.array(p['gains'])[None,:,None]).reshape(r.shape); print('raw vs fitted vs payload norms',*[np.linalg.norm(t,axis=0) for t in (r,y,a)]); print('calibrated readout minmax by pulse',y.min(0),y.max(0))
for name,q in [('open',[1]*8),('sealed',[1,1,0,0]*2)]:
 pred=matrices(s,q)[0]@w
 for tag,t in [('raw',r),('staged',y),('payload',a)]:
  e=t-pred; print(name,tag,'rel',np.linalg.norm(e,axis=0)/np.linalg.norm(t,axis=0),'Z residual',np.linalg.norm(e-s['Z_nuisance']@np.linalg.lstsq(s['Z_nuisance'],e,rcond=None)[0],axis=0),'channel RMSE',np.mean(e.reshape(-1,4,3)**2,axis=0)**.5)
 print('pred vs calibrated by shift first pulse',np.stack([pred[:,0],y[:,0]],axis=1).reshape(3,32,2)[:,:8,:])
for tag,t in [('staged',y),('payload',a)]:
 for bounds in [([.01,.01,-2,-2]*2,[10,10,5,5]*2)]:
  fit=least_squares(lambda q:(matrices(s,q)[0]@w-t).ravel(),[1,1,.02,.02]*2,bounds=bounds,max_nfev=160)
  print('signed stencil',tag,fit.x,np.linalg.norm(fit.fun.reshape(-1,3),axis=0)/np.linalg.norm(t,axis=0))
