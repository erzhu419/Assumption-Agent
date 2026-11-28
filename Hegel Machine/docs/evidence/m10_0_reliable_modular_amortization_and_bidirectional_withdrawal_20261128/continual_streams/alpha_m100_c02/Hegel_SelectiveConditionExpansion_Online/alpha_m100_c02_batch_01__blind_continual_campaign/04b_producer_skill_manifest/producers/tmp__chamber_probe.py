import sys,pickle,json,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m100_c02_s00_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te,g=d['train'],d['test'],p['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; np.set_printoptions(precision=5,suppress=True)
out=m.solve_batch(d); tau,th=out['estimated_taus'],out['hereditary_weights']; print('material',tau,th)
for q,x in [(tr,m._features(tr,te)[:len(tr['lab_id'])]),(g,m._features(tr,g)[len(tr['lab_id']):])]:
 past,hold=m._loads(cal,q); a,b,*_=m._design(np.log(tau),past,hold,x,dt,t); y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; e=y-a@th; f=r-b@th; print('base err rmse/median',np.sqrt(np.mean(e**2)),np.median(abs(e)),np.sqrt(np.mean(f**2)),np.median(abs(f))); print('median err by time',np.median(e,axis=0)); print('stress bounds',past.min(),past.max(),hold.min(),hold.max()); print('resid correlations',np.corrcoef(np.c_[past,hold,x[:,1:],f,e[:,-1]].T)[-2:,:-2]);
 F=np.c_[-np.expm1(-t[:,None]/tau),t]; A=np.vstack([F,np.r_[1/tau,1]]); coeff=np.linalg.lstsq(A,np.c_[y,r].T,rcond=None)[0].T; print('free coupon amplitude median shape error',np.median(abs(np.c_[y,r]-coeff@A.T))); print('amp delta first8', (coeff-np.c_[b[:,:2]*th[:2]*tau,x@th[2:]])[:8]); print('baseline amp first8',np.c_[b[:,:2]*th[:2]*tau,x@th[2:]][:8]); print('missing',np.isnan(q['coupon_features_raw']).sum(0))
for record in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(record['facility_telemetry_trace']); print(record['completed_campaign_short_name'],a.mean(0),a.std(0))
