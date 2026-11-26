import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_w02_batch_07_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
case=pickle.load(open('/holdout_task/public_alpha_w02_batch_07.pkl','rb'))[0]
d=case['dataset']; out=m.solve_alpha_w02_batch_07(d); g=case['gt']; c=d['calibration_load_cells']; cal=m._calibrate(c,d['yield_stress']); print('calibration',cal.tolist())
print('cal max errors',np.max(abs(cal-np.c_[g['stress_gains'],g['stress_biases']]),axis=0))
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
 print(a,'estimate',out[a],'target',g[b],'max_abs',max(abs(out[a]-g[b])))
t=d['test']; phys=np.c_[g['te_hist_true'],g['te_hold_true']]; sub=(abs(phys)<=d['yield_stress']).all(axis=1); complete=np.isfinite(t['coupon_features_raw']).all(1)
e=out['predicted_strain']-g['te_clean']; er=out['initial_rate_pred']-g['te_true_rate0']; rmse=lambda v: np.sqrt(np.mean(v*v))
for label,mask in [('all',np.ones(len(e),bool)),*[(f'lab {i}',t['lab_id']==i) for i in range(4)],('subyield',sub),('overstress',~sub),('complete',complete),('missing',~complete)]:
 if mask.any(): print(label,mask.sum(),'curve and onset RMSE',rmse(e[mask]),rmse(er[mask]),'early/late',rmse(e[mask,0]),rmse(e[mask,-1]))
print('observed test rate RMSE',rmse(t['instant_rate_obs']-g['te_true_rate0']))
