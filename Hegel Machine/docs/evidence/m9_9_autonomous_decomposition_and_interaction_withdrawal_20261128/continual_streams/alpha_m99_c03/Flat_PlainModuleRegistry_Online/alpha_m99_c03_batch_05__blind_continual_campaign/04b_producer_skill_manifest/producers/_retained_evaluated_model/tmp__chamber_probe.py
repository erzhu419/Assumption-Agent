import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c03_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; d=item['campaign_diagnostics']; tr=ds['train']; te=ds['test']; t=ds['eval_times']; dt=ds['past_dt']; n=len(tr['lab_id']); p=len(te['lab_id'])
keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); both={k:np.concatenate((te[k],d[k])) for k in keys}
x=m._features(tr,te); xd=m._features(tr,both)[n+p:]; xd2=m._features(tr,d)[n:]; co=m._calibration(ds['calibration_load_cells']); inp=m._coupon_inputs(tr,x[:n],co); inp_d=m._coupon_inputs(d,xd,co)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; yd=d['strain_curves_obs']-d['strain_0'][:,None]; rd=d['instant_rate_obs']; tau,th=m._fit(inp,dt,t,y,r)
def score(e): return np.round([np.median(abs(e)),np.sqrt(np.mean(e**2)),np.max(abs(e))],6).tolist()
if __name__=='__main__':
 print('material',tau,th,'diagnostic feature delta',np.max(abs(xd-xd2)),'missing',np.isnan(d['coupon_features_raw']).sum(axis=0))
 for label,b,xx,obs,rate in [('train',inp,x[:n],y,r),('diagnostics',inp_d,xd,yd,rd)]:
  a,g=m._design(tau,b,dt,t); e=obs-a@th; er=rate-g@th
  print(label,'inputs',[(v.shape,np.round(np.percentile(v,[0,50,100]),4).tolist()) for v in b], 'curve',score(e),'rate',score(er)); print('curve median by horizon',np.median(e,axis=0).round(5),'rate median',np.median(er).round(5))
  if label=='diagnostics':
   print('lab/stratum',np.unique(np.c_[d['lab_id'],d['stratum']],axis=0,return_counts=True)); print('first 10 rate residuals and curve residuals at endpoints',np.c_[er,e[:,0],e[:,-1]][:10].round(5)); print('implied OLS weights',np.linalg.lstsq(np.vstack([a.reshape(-1,7),g]),np.r_[obs.ravel(),rate],rcond=None)[0]); print('aux train/test percentiles',*(np.percentile(v['aux_chamber_telemetry'],[0,50,100],axis=0).round(4).tolist() for v in (tr,te)))
 print('nonzero bridge terms',[(j,[('bias' if i==0 else m._TERMS[i-1],round(v,4)) for i,v in enumerate(co[:,j]) if v]) for j in range(4)])
