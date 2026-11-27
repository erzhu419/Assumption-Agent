import importlib.util, pickle, time, numpy as np
p='/holdout_task/alpha_m96_w01_s00_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
c=pickle.load(open('/holdout_task/public_alpha_m96_w01_s00.pkl','rb'))[0]; d=c['dataset']; gt=c['gt']; te=d['test']; start=time.perf_counter(); o=m.solve_alpha_m96_w01_s00(d); print('seconds',time.perf_counter()-start)
for name,key in [('estimated_taus','taus'),('hereditary_weights','theta')]: print(name,o[name], 'error',o[name]-gt[key])
e=o['predicted_strain']-gt['te_clean']; r=o['initial_rate_pred']-gt['te_true_rate0']; noise=te['instant_rate_obs']-gt['te_true_rate0']; missing=~np.isfinite(te['coupon_features_raw']).all(1); rmse=lambda v: np.sqrt(np.mean(v*v))
for name,mask in [('all',np.ones(len(e),bool)),('missing',missing),('complete',~missing)]+[(f'lab{i}',te['lab_id']==i) for i in range(4)]: print(name,'curve/rate RMSE',rmse(e[mask]),rmse(r[mask]),'rate measurement RMSE',rmse(noise[mask]),'rate-corrected curve RMSE',rmse((e+(noise-r)[:,None]*d['eval_times'])[mask]))
print('horizon RMSE',np.sqrt(np.mean(e*e,axis=0)))
