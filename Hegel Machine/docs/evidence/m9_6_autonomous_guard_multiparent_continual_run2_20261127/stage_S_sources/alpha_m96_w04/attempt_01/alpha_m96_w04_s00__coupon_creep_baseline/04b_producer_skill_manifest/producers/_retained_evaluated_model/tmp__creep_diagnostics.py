import pickle, sys, time
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m96_w04_s00_solver as s
with open('/holdout_task/public_alpha_m96_w04_s00.pkl','rb') as f: case=pickle.load(f)[0]
print('case keys',list(case)); d=case['dataset']; print('test keys',list(d['test']))
c=s._calibration(d['calibration_load_cells']); print('calibration',c)
cal=d['calibration_load_cells']
for lab in range(4):
 m=cal['lab_id']==lab
 r=cal['true_stress'][m]-c[lab,0]*cal['cmd_stress'][m]-c[lab,1]
 print('cal residual',lab,np.round(np.sort(np.abs(r)),7))
t=time.perf_counter(); out=s.solve_alpha_m96_w04_s00(d); print('seconds',time.perf_counter()-t)
print('taus',out['estimated_taus'],'weights',out['hereditary_weights'])
for part in ('train','test'):
 p=d[part]; n=len(d['train']['lab_id']); x=s._aligned((d['train'],d['test'])); x=x[:n] if part=='train' else x[n:]
 model=s._Model(p,x,c,d['past_dt'],d['eval_times']); pred=model.basis(out['estimated_taus'])[0] @ out['hereditary_weights']; print(part,'rate abs q',np.quantile(abs(pred[:,-1]-p['instant_rate_obs']),[.5,.8,.9,.95,.99,1]))
 if 'strain_curves_obs' in p:
  r=pred[:,:-1]+p['strain_0'][:,None]-p['strain_curves_obs']; print(part,'curve abs q',np.quantile(abs(r),[.5,.8,.9,.95,.99,1]),'mad_horizon',s._mad(r,axis=0))
  for mask in [np.isfinite(p['coupon_features_raw']).all(1),~np.isfinite(p['coupon_features_raw']).all(1)]: print('missingness slice',mask.sum(),np.median(abs(r[mask])),np.median(abs(pred[mask,-1]-p['instant_rate_obs'][mask])))
