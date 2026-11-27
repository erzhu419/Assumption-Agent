import glob, importlib.util, json, pickle
import numpy as np
p=glob.glob('/tmp/hegel_continual_archive/*batch_02_solver.py')[0]
s=importlib.util.spec_from_file_location('base',p); b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']
xtr,xte=b._standardize(tr,te); stress=b._stress_calibration(cal)
a=b._design(tr,xtr,stress,dt,t); tau,beta=b._fit(a,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('dt',dt,'times',t,'tau',tau,'beta',beta)
for name,c in [('train',tr),('diag',q)]:
 x=xtr if name=='train' else b._standardize(dict((k,np.concatenate((tr[k],te[k]))) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']),q)[1]
 S,R=b._design(c,x,stress,dt,t)(tau); e=c['strain_curves_obs']-c['strain_0'][:,None]-S@beta; er=c['instant_rate_obs']-R@beta
 print(name,'strain med abs by time',np.median(abs(e),0).round(6),'rate med abs',np.median(abs(er)),'rate residuals',er.round(5))
 if name=='diag':
  for j in range(len(er)): print(j,c['lab_id'][j],c['stratum'][j],np.round(e[j,[0,2,5,8,11]],5),round(er[j],5),round(stress(c['hold_channel_raw'])[j],3),np.round(stress(c['past_channel_raw'])[j],2))
raw=cal['bridge_raw']; true=cal['true_channel_loads']; E=np.column_stack((np.ones(len(raw)),raw)); err=true-E@np.linalg.lstsq(E,true,rcond=None)[0]; print('calibration RMSE',np.mean(err**2,axis=0)**.5,'total calibration RMSE',np.mean(err.sum(1)**2)**.5,'load std',np.std(true,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
print('other keys',list(d),list(tr),list(te)); logs=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('telemetry schema',logs[0]['facility_telemetry_trace'][0]);
for l in logs:
 a=np.asarray(l['facility_telemetry_trace']); print(l['completed_batch_index'],a.shape,'mean',a.mean(0).round(4),'sd',a.std(0).round(4),'head',a[:2].round(4))
