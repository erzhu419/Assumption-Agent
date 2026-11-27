import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb') as f: a=pickle.load(f)[0]
d,g=a['dataset'],a['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); out=m.solve_alpha_m97_c01_batch_02(d)
past,hold,x,z=m._inputs(d); dt,t=d['past_dt'],d['eval_times']; missing=np.isnan(te['coupon_features_raw']).any(1)
assert np.array_equal(out['regime_mode_weights'],np.zeros(2))
for k,h in [('estimated_taus','taus'),('hereditary_weights','theta')]:
 print(k,'max abs error',np.max(np.abs(out[k]-g[h])))
for k,h in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
 e=np.abs(out[k]-g[h]); print(k,'overall MAE/RMSE/max',e.mean(),np.sqrt(np.mean(e**2)),e.max())
 print('  each lab MAE',[e[te['lab_id']==i].mean() for i in range(4)],'complete/missing',[e[~missing].mean(),e[missing].mean()])
 if e.ndim==2: print('  horizon MAE',e.mean(0))
print('physical stress max errors',np.max(np.abs(past[n:]-g['te_hist_true'])),np.max(np.abs(hold[n:]-g['te_hold_true'])))
y=np.column_stack((tr['strain_curves_obs']-z[:n,None],tr['instant_rate_obs']))
for i in range(4):
 idx=tr['lab_id']==i
 def model(mask): return m._Model(dt,t,past[:n][mask],hold[:n][mask],x[:n][mask])
 lt,w=m._fit(model(~idx),y[~idx]); err=np.abs(model(idx).design(lt)[0]@w-y[idx]); print('held-out lab',i,'curve/rate median abs errors',np.median(err[:,:-1]),np.median(err[:,-1]))
# Compare the closed-form history and analytic initial rate with the literal recurrence.
tau,hw=out['estimated_taus'],out['hereditary_weights']; memory=np.zeros((len(hold),2))
for j,duration in enumerate(dt):
 decay=np.exp(-duration/tau); memory=decay*memory+(1-decay)*past[:,j,None]
gaps=hold[n:,None]-memory[n:]; drift=hw[2]+x[n:]@hw[3:]; assert np.allclose(out['initial_rate_pred'],gaps@(hw[:2]/tau)+drift,atol=1e-11,rtol=0)
assert np.allclose(out['predicted_strain'],z[n:,None]+(gaps*hw[:2])@(-np.expm1(-t[None,:]/tau[:,None]))+drift[:,None]*t,atol=1e-11,rtol=0)
