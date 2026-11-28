import pickle, runpy, numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; data=item['dataset']; tr,te,di=data['train'],data['test'],item['campaign_diagnostics']; dt,t=data['past_dt'],data['eval_times']; out=m['solve_batch'](data); tau,th=(out[k] for k in ('estimated_taus','hereditary_weights')); B=m['_bridge'](data['calibration_load_cells']); xa,xb=m['_features'](tr,te); xd=m['_features'](di,di)[0]; S=m['_prepare'](di,xd,B); y=np.column_stack((di['instant_rate_obs'],di['strain_curves_obs']-di['strain_0'][:,None])); w=np.r_[.45,np.ones(len(t))];
def pack(cr):return np.column_stack((cr[1],cr[0]))
zero=np.zeros(2); b=pack(m['_chamber'](S,tau,th,t,zero)); A=np.stack([pack(m['_chamber'](S,tau,th,t,f))-b for f in np.eye(2)],axis=2)
def fit(j): return np.linalg.lstsq((A[j]*w[None,:,None]).reshape(-1,2),((y-b)[j]*w).ravel(),rcond=None)[0]
print('material',tau,th, 'fractions',fit(np.ones(len(y),bool)))
for labels in [np.arange(len(y)),di['lab_id'],di['stratum']]:
 metrics=[]
 for l in np.unique(labels):
  i=labels==l; e=(y-b-A@fit(~i))[i]; metrics.append([np.sqrt(np.mean(e[:,0]**2)),np.sqrt(np.mean(e[:,1:]**2)),np.sqrt(np.mean(e[:,-1]**2))]);
 print('worst CV rate, curve, final',np.max(metrics,axis=0))
e=y-pack(m['_chamber'](S,tau,th,t)); print('candidate chamber rate, curve, final RMSE',np.sqrt(np.mean(e[:,0]**2)),np.sqrt(np.mean(e[:,1:]**2)),np.sqrt(np.mean(e[:,-1]**2))); print('deployment drift range',np.min(th[2]+xb@th[3:]),np.max(th[2]+xb@th[3:])); print('diagnostic residual correlation with drift',((S[2]@th[2:])@e)/np.sum((S[2]@th[2:])**2))
