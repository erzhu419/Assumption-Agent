import pickle,numpy as np
from base import _features,_bridge,_inputs,_design
np.set_printoptions(precision=5,suppress=True,linewidth=140)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=a['dataset']; tr,te,q=d['train'],d['test'],a['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; b=_bridge(c['bridge_raw'],c['true_channel_loads']); m=np.load('/tmp/material.npz'); T,W,x=[m[k] for k in ('T','W','x')]
p,h,F=_inputs(q,x,b); C,R=_design(T,p,h,F,dt,t); y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; pred=np.c_[C@W,R@W]; e=y-pred
A=np.vstack((np.c_[-np.expm1(-t[:,None]/T),t],np.r_[1/T,1])); v=e@np.linalg.pinv(A).T
aux=np.empty((len(p),0)); names=['p'+str(i) for i in range(p.shape[1])]+['h']+['x'+str(i) for i in range(4)]; X=np.c_[p,h,x]
if __name__=='__main__':
 for s in (tr,te,q):
  print('split',len(s['lab_id']),'missing',np.isnan(s['coupon_features_raw']).sum(0),'telemetry quantiles',np.quantile(s['aux_chamber_telemetry'],[0,.5,1],axis=0))
 print('svals error',np.linalg.svd(e,compute_uv=False)); print('svals residual after temporal fit',np.linalg.svd(e-v@A.T,compute_uv=False)); print('corr'); print([(n,np.corrcoef(np.c_[X[:,i],v].T)[0,1:].round(3).tolist()) for i,n in enumerate(names)]); print('coeff',np.linalg.lstsq(np.c_[np.ones(len(X)),X],v,rcond=None)[0]); print('stress and telem'); print(np.c_[p,h,aux]);
