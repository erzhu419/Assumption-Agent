import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c02_batch_01_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d=i['dataset']; tr=d['train']; n=len(tr['lab_id']); cal=m._calibration(d); x=m._align(d); args=m._inputs(d,tr,cal,x[:n]); yt=tr['strain_curves_obs']-tr['strain_0'][:,None]; rt=tr['instant_rate_obs']; labs=tr['lab_id']
def subset(a,mask): return tuple(v if j in (2,3) else v[mask] for j,v in enumerate(a))
def err(v): return np.round([np.median(np.abs(v)),np.quantile(np.abs(v),.9),np.sqrt(np.mean(v**2))],5)
for mode in ('gated','all_log'):
 a=args if mode=='gated' else args[:-1]+(np.ones(n,bool),)
 print('MODE',mode,flush=True)
 for lab in [-1,0,1,2,3]:
  fit=labs!=lab; val=~fit if lab>=0 else fit
  p=m._fit(subset(a,fit),yt[fit],rt[fit]); c,r,*_=m._design(p[:2],*subset(a,val)); ec=c@p[2:]-yt[val]; er=r@p[2:]-rt[val]; ac=args[-1][val]; print(lab,np.round(np.r_[np.exp(p[:2]),p[2:]],5),[(s,err(ec[ac==s]),err(er[ac==s])) for s in (False,True) if np.any(ac==s)],flush=True)
out=m.solve_alpha_m97_c02_batch_01(d); gt=i['gt']; print('public parameter errors',[(k,np.asarray(out[k])-np.asarray(gt[g])) for k,g in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]]); print('public clean error',err(out['predicted_strain']-gt['te_clean']),err(out['initial_rate_pred']-gt['te_true_rate0']))
