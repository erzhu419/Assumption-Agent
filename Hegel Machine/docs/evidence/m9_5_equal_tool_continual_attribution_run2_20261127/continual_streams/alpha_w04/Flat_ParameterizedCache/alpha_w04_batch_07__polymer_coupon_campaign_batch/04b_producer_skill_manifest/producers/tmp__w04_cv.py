import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w04_batch_07_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for case,record in enumerate(pickle.load(open('/holdout_task/public_alpha_w04_batch_07.pkl','rb'))):
 D=record['dataset']; tr,te=D['train'],D['test']; lab=tr['lab_id']; t,dt=D['eval_times'],D['past_dt']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; missing=~np.isfinite(tr['coupon_features_raw']); panel=D['calibration_load_cells']; base=m._calibrate(panel); pair=base.copy()
 for j in (2,3):
  u,v=m._rows(panel,j); pair[j]=m._line(u[:2],v[:2]) if len(u)>=2 else base[j]
 print('case',case,'counts',np.bincount(lab),'missing',np.bincount(lab[missing.any(1)],minlength=4),'affine',base.tolist(),'plastic state',[(k,int(np.count_nonzero(v['residual_plastic_state']))) for k,v in [('train',tr),('test',te)] if 'residual_plastic_state' in v],flush=True)
 for name,a,complete in [('archive',base,False),('archive-complete',base,True),('first-pair',pair,False)]:
  ep=np.empty_like(y); er=np.empty_like(r); ec=np.empty_like(y); fits=[]
  for j in range(4):
   val=lab==j; train=~val & (~missing.any(1) if complete else True)
   subset=lambda z:{k:np.asarray(v)[z] for k,v in tr.items()}
   f,g=subset(train),subset(val); x,xt=m._align(f,g)
   st=lambda v,z:m._prepare(dt,*m._stress(v,a),z,D['yield_stress']); tau,w=m._fit(t,st(f,x),y[train],r[train]); P,R=m._design(tau,t,st(g,xt)); ep[val]=P@w-y[val]; er[val]=R@w-r[val]; lim=3*np.std((x-x.mean(0))@(missing[val]*w[3:7]).T,axis=0); ec[val]=ep[val]+np.clip(-er[val],-lim,lim)[:,None]*t; fits.append(tau)
  def stats(p,q,z): return np.round([np.sqrt(np.mean(p[z]**2)),np.median(abs(p[z])),np.sqrt(np.mean(q[z]**2)),np.median(abs(q[z]))],5).tolist()
  print(name,'tau range',np.round([np.min(fits,0),np.max(fits,0)],5).tolist(),'by lab [curve RMSE, curve medAE, rate RMSE, rate medAE]',[stats(ep,er,lab==j) for j in range(4)],'missing raw/corrected',[(stats(ep,er,(lab==j)&missing.any(1))[:2],stats(ec,er,(lab==j)&missing.any(1))[:2]) for j in range(4) if np.any((lab==j)&missing.any(1))],flush=True)
