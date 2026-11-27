import pickle,runpy,numpy as np
s=runpy.run_path('/holdout_task/alpha_m96_w03_s00_solver.py')
e,=pickle.load(open('/holdout_task/public_alpha_m96_w03_s00.pkl','rb'))
d=e['dataset']; tr,te=d['train'],d['test']; gt=e['gt']; dt,t=d['past_dt'],d['eval_times']; x,xe=s['_align'](tr,te); rmse=lambda a: np.sqrt(np.mean(a*a)); cal=s['_calibrate'](d['calibration_load_cells']); complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for mask in [np.ones(len(x),bool),complete]:
 b={k:v[mask] for k,v in tr.items()}; p=s['_fit'](s['_Hold'](dt,t,b,x[mask],cal),b); h=s['_Hold'](dt,t,te,xe,cal); y=h.design(p[:2])[0] @ p[2:]; m=np.isfinite(te['coupon_features_raw']).all(axis=1)
 print('public fit rows',sum(mask),'tau error',p[:2]-gt['taus'],'theta RMSE',rmse(p[2:]-gt['theta']),'complete strain/rate RMSE',rmse((y[:,:-1]+h.base[:,None]-gt['te_clean'])[m]),rmse((y[:,-1]-gt['te_true_rate0'])[m]))
rng=np.random.default_rng(612); gains=np.array([.94,1.11,.87,1.08]); biases=np.array([.11,-.16,.21,-.03]); c0=d['calibration_load_cells']; lab=c0['lab_id']; clean=gains[lab]*c0['cmd_stress']+biases[lab]; c={**c0,'true_stress':clean+rng.normal(0,8e-5,len(lab))}
for i in [2,3]:
 idx=np.flatnonzero(lab==i); pick=idx[np.argsort(np.abs(c['cmd_stress'][idx]))[-2:]]; c['true_stress'][pick]+=[.8,-1.2]
g,b=s['_calibrate'](c); print('two EMI spikes per affected lab gain and bias absolute error:',abs(g-gains),abs(b-biases));
for tau in [[.2,8.],[.38,2.],[1.2,6.5],[2.4,7.9]]:
 def physical(block,xx):
  ids=block['lab_id']; source={**block,'past_stress_cmd':(block['past_stress_cmd']-biases[ids,None])/gains[ids,None], 'stress_hold_cmd':(block['stress_hold_cmd']-biases[ids])/gains[ids]}; patched=s['_Hold'](dt,t,source,xx,(gains,biases)); cols=[]
  for u in tau:
   z=np.zeros(len(ids))
   for j,step in enumerate(dt):
    a=np.exp(-step/u); z=a*z+(1-a)*block['past_stress_cmd'][:,j]
   delta=block['stress_hold_cmd']-z; cols.append(delta[:,None]*np.r_[1-np.exp(-t/u),1/u])
  mat=np.concatenate((np.stack(cols,axis=-1),patched.linear),axis=-1); np.testing.assert_allclose(mat,patched.design(tau)[0],atol=2e-14,rtol=0); return source,mat @ gt['theta']
 a,ya=physical(tr,x); z,yz=physical(te,xe); a={**a,'instant_rate_obs':ya[:,-1]+rng.normal(0,.012,len(ya))}; noise=rng.normal(0,.01,ya[:,:-1].shape); glitch=rng.random(noise.shape)<.08; noise[glitch]+=rng.standard_cauchy(glitch.sum())*.4; a['strain_curves_obs']=ya[:,:-1]+a['strain_0'][:,None]+noise
 p=s['solve_alpha_m96_w03_s00']({**d,'train':a,'test':z,'calibration_load_cells':c}); ds=p['predicted_strain']-z['strain_0'][:,None]-yz[:,:-1]; dr=p['initial_rate_pred']-yz[:,-1]; print('stress tau',tau,'fit',p['estimated_taus'],'theta RMSE',rmse(p['hereditary_weights']-gt['theta']),'worst-lab strain/rate RMSE',max(rmse(ds[z['lab_id']==i]) for i in range(4)),max(rmse(dr[z['lab_id']==i]) for i in range(4)))
