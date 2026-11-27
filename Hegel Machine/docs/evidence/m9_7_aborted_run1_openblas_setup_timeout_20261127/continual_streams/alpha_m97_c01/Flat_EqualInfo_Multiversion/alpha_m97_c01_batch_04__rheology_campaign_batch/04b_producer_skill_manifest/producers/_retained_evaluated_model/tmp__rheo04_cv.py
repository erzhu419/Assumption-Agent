import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',P+'alpha_m97_c01_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open(P+'public_alpha_m97_c01_batch_04.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; state=m._prepare(d)
print('calibration',m._calibrate(d['calibration_load_cells'])); print('public reference schema',{k:np.shape(v) for k,v in item['gt'].items()})
def rm(e): return np.sqrt(np.mean(e**2))
def stats(label,ec,er): print(label,'curve MAE/RMSE',np.mean(np.abs(ec)),rm(ec),'rate MAE/RMSE',np.mean(np.abs(er)),rm(er),'late RMSE',rm(ec[:,-1]))
lab=tr['lab_id']; ec=np.empty_like(y); er=np.empty_like(r)
for c in np.unique(lab):
 a=lab!=c; b=~a
 tau,coef=m._fit(tuple(v[:n][a] for v in state),dt,t,y[a],r[a]); A,B=m._design(tau,tuple(v[:n][b] for v in state),dt,t)
 ec[b]=A@coef-y[b]; er[b]=B@coef-r[b]; stats(('held lab',c,'tau',np.round(tau,5)),ec[b],er[b])
for b,label in [(state[4][:n]>0,'active'),(state[4][:n]==0,'zero damage'),(np.isnan(tr['coupon_features_raw']).any(axis=1),'missing coupon'),(np.ones(n,dtype=bool),'pooled')]:
 if b.any(): stats((label,b.sum()),ec[b],er[b])
out=m.solve_alpha_m97_c01_batch_04(d)
for key in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(key,out[key])
gt=item['gt']
if 'te_clean' in gt: stats('public references',out['predicted_strain']-gt['te_clean'],out['initial_rate_pred']-gt['te_true_rate0'])
print('test activation count',(state[4][n:]>0).sum(),'/',len(te['lab_id']))
