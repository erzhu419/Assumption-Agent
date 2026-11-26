import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
e=pickle.load(open('/holdout_task/public_alpha_w01_batch_01.pkl','rb'))[0]; d,g=e['dataset'],e['gt']; q=d['test']; o=m.solve_alpha_w01_batch_01(d)
print('taus predicted vs public',o['estimated_taus'],g['taus'])
for k,kg in [('predicted_strain','te_clean'),('secant_compliance','te_secant'),('hazard_crossing_times','te_cross')]:
 err=np.asarray(o[k])-g[kg]; rms=lambda a: np.sqrt(np.mean(a*a)); print(k,'RMSE',rms(err),'max',np.max(abs(err)), 'labs',[round(rms(err[q['lab_id']==i]),7) for i in range(4)])
 if k=='hazard_crossing_times':
  hit=g[kg]<d['eval_times'][-1]; print('miss/false',sum(hit & (o[k]==d['eval_times'][-1])),sum(~hit & (o[k]<d['eval_times'][-1])), 'signed hit quantiles', np.quantile(err[hit],[0,.1,.5,.9,1]))
print('missing features test/train',np.where(~np.isfinite(q['coupon_features_raw']).all(axis=1))[0],sum(~np.isfinite(d['train']['coupon_features_raw']).all(axis=1)))
