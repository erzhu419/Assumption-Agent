import importlib.util
import pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('s',p+'alpha_m97_c03_batch_07_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
record=pickle.load(open(p+'public_alpha_m97_c03_batch_07.pkl','rb'))[0]; d,g=record['dataset'],record['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; x,z=s._align(tr,te); lines=s._calibrate(d['calibration_load_cells'])
args=(lines,dt,d['crit_stress'],True); inputs=s._inputs(tr,x,*args); test=s._inputs(te,z,*args)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); test_complete=np.isfinite(te['coupon_features_raw']).all(axis=1)
def report(e, m):
 return np.round([np.sqrt(np.mean(e[m,:-1]**2)),np.sqrt(np.mean(e[m,-1]**2)),np.median(abs(e[m,:-1])),np.median(abs(e[m,-1]))],6)
for mode in (None,complete):
 print('complete only' if mode is not None else 'all coupons')
 out=s._design(np.log([.55,3.5]),inputs,dt,t,True)[:,:,0].copy()
 for lab in range(4):
  take=tr['lab_id']!=lab; hold=~take
  b,w=s._fit(tuple(a[take] for a in inputs),dt,t,y[take],True,mode[take] if mode is not None else None)
  out[hold]=s._design(b,tuple(a[hold] for a in inputs),dt,t,True)@w
 print('LOLO by lab [curve RMSE, rate RMSE, curve MedAE, rate MedAE]:', [report(out-y,(tr['lab_id']==lab)&complete).tolist() for lab in range(4)])
 print('LOLO by activation:',[report(out-y,complete&((inputs[-1]>0)==a)).tolist() for a in (False,True)])
 b,w=s._fit(inputs,dt,t,y,True,mode)
 print('parameter max errors tau, hereditary, regime:',np.max(abs(np.exp(b)-g['taus'])),np.max(abs(w[:7]-g['theta'])),np.max(abs(w[7:]-g['vp_theta'])))
 e=(s._design(b,test,dt,t,True)@w)-np.c_[g['te_clean']-te['strain_0'][:,None],g['te_true_rate0']]
 print('public test by lab all:',[report(e,te['lab_id']==lab).tolist() for lab in range(4)])
 print('public test complete / imputed:',report(e,test_complete),report(e,~test_complete),'worst horizon RMSE',np.sqrt(np.max(np.mean(e[:,:-1]**2,axis=0))))
