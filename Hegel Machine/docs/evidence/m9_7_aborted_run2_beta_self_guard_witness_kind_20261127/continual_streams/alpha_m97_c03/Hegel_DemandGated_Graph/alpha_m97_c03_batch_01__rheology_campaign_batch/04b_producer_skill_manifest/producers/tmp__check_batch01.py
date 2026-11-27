import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_m97_c03_batch_01_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open(p+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]; d,g=i['dataset'],i['gt']; o=m.solve_alpha_m97_c03_batch_01(d)
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]:
 print(k,'fit',o[k],'max abs error',np.max(abs(o[k]-g[j])))
lab=d['test']['lab_id']; act=(np.maximum(abs(g['te_hist_true'])-d['crit_stress'],0)**2 @ d['past_dt'])>0
missing=np.any(~np.isfinite(d['test']['coupon_features_raw']),axis=1)
ec=o['predicted_strain']-g['te_clean']; er=o['initial_rate_pred']-g['te_true_rate0']; rm=lambda e:np.sqrt(np.mean(e*e))
for name,use in [(f'lab {l} active {a}',(lab==l)&(act==a)) for l in range(4) for a in (False,True)]+[(f'missing {v}',missing==v) for v in (False,True)]:
 print(name,'n',sum(use),'curve/rate RMSE',rm(ec[use]),rm(er[use]))
tr,te=d['train'],d['test']; x,_=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); c,r=m._design(tr,x,cal,d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index'])(np.log(o['estimated_taus'])); b=np.r_[o['hereditary_weights'],o['regime_mode_weights']]
for name,e in [('curve',c@b+tr['strain_0'][:,None]-tr['strain_curves_obs']),('rate',r@b-tr['instant_rate_obs'])]:
 mask=np.any(~np.isfinite(tr['coupon_features_raw']),axis=1)
 for missing_rows in (False,True):
  z=e[mask==missing_rows]; print('train',name,'missing',missing_rows,'abs residual q50/q90/q99/max',np.quantile(abs(z),[.5,.9,.99,1]))
