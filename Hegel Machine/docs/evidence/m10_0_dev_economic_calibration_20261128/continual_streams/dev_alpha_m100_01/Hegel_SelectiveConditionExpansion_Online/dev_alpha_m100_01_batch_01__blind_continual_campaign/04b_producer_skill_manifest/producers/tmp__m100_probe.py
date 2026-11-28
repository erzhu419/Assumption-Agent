import pickle,numpy as np
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=130)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; diag=i['campaign_diagnostics']; tr=d['train']; te=d['test']; cal=d['calibration_load_cells']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),diag)[-len(diag['lab_id']):]; b=_bridge(cal)
a=cal['bridge_raw']; y=cal['true_channel_loads']; z=np.c_[a,np.ones(len(a))]; e=z@np.linalg.lstsq(z,y,rcond=None)[0]-y
print('cal',a.shape,'ranges',a.min(0),a.max(0),'affine RMSE',np.sqrt(np.mean(e**2,0)),'max',np.max(abs(e),0),'sweeps',type(cal['single_channel_sweeps'])); print('sweep keys',[(str(k),getattr(v,'shape',None),list(v) if isinstance(v,dict) else '') for k,v in cal['single_channel_sweeps'].items()] if isinstance(cal['single_channel_sweeps'],dict) else str(cal['single_channel_sweeps'])[:350]); print('bridge',b,'dt,t',dt,t)
for typ in ('train','diag'):
 part=tr if typ=='train' else diag; xx=x[:n] if typ=='train' else xd; p,h=_loads(part,b); cobs=part['strain_curves_obs']-part['strain_0'][:,None]; robs=part['instant_rate_obs']; fit=_fit(p,h,xx,dt,t,cobs,robs)
 print('fit',typ,fit)
 if typ=='train': tau,theta=fit
 C,R=_design(tau,p,h,xx,dt,t); ec=cobs-C@theta; er=robs-R@theta
 print(typ,'resid quantiles curve',np.quantile(abs(ec),[.25,.5,.8,.9,.95,1]),'rate',np.quantile(abs(er),[.25,.5,.8,.9,.95,1])); print('resid means by t',ec.mean(0)); print('curve SVD',np.linalg.svd(ec,compute_uv=False)); print('resid first rows',np.c_[er,ec][:8]); print('part lab,str counts',np.unique(np.c_[part['lab_id'],part['stratum']],axis=0,return_counts=True));
 if typ=='diag': print('hold/past ranges',h.min(),h.max(),p.min(),p.max())
