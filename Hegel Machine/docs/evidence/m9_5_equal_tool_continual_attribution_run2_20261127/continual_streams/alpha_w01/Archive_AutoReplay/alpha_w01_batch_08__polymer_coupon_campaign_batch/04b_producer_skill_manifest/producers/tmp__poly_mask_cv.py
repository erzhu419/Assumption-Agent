import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_batch_08_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
o=pickle.load(open('/holdout_task/public_alpha_w01_batch_08.pkl','rb'))[0]; d=o['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; labs=tr['lab_id']; obs=tr['instant_rate_obs']; x,z=m._alignment(tr,te); maps=m._load_cells(d['calibration_load_cells']); inp=m._inputs(tr,x,maps,dt,d['yield_stress']); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; ids=np.flatnonzero(np.isfinite(tr['coupon_features_raw']).all(1)); miss=~np.isfinite(te['coupon_features_raw']); patterns=np.unique(miss[miss.any(1)],axis=0); late=t>=np.median(t)
fold=np.zeros(len(x),int); rng=np.random.default_rng(421)
for lab in range(4):
 idx=ids[labs[ids]==lab].copy(); rng.shuffle(idx); fold[idx]=np.arange(len(idx))%5
err={}
for f in range(5):
 a=ids[fold[ids]!=f]; b=ids[fold[ids]==f]; A={k:tr[k][a] for k in ('strain_curves_obs','strain_0','instant_rate_obs')}; U=tuple(v[a] for v in inp); tau,w=m._fit(A,U,dt,t); c=m._lab_drift(A,labs[a],U,tau,w,dt,t)
 def correction(idx): return c[labs[idx],0]+c[labs[idx],1]*(x[idx]@w[3:7])
 _,ra=m._design(tau,U,dt,t); ce,re=m._design(tau,tuple(v[b] for v in inp),dt,t); residual=obs[a]-ra@w-correction(a); noise=(1.4826*np.median(abs(residual-np.median(residual))))**2; p=re@w+correction(b); target=p+np.median((y[b]-ce@w-correction(b)[:,None]*t)[:,late]/t[late],axis=1)
 for lab in range(4):
  i=a[labs[a]==lab]; bm=labs[b]==lab; q=b[bm]; weights=w[3:7]*(1+c[lab,1]); e=residual[labs[a]==lab]; nl=(1.4826*np.median(abs(e-np.median(e))))**2
  for j,h in enumerate(patterns):
   latent=x[i][:,h]@weights[h]; base=p[bm]-x[q][:,h]@weights[h]; global_fill=np.array([z[miss[:,k],k][0] for k in np.flatnonzero(h)])@weights[h]; v0=np.var(x[a][:,h]@weights[h]); known=~h; X=x[i][:,known]; center=X.mean(0); C=X-center; beta=np.linalg.solve(C.T@C+np.diag(np.var(C,axis=0)*len(i)*.25+1e-9),C.T@(latent-latent.mean())); mu=latent.mean()+(x[q][:,known]-center)@beta; v=np.var(latent-C@beta)
   options=[(global_fill,v0,noise),(latent.mean(),np.var(latent),noise),(mu,v,noise),(mu,v,nl),(0,1,0)]
   err.setdefault((lab,j),[]).append(np.array([base+s+v/(v+n)*(obs[q]-base-s)-target[bm] for s,v,n in options]))
print('Columns: current, lab mean, conditional, conditional with lab noise, onset only')
for key, values in sorted(err.items()): print(key,np.sqrt(np.mean(np.concatenate(values,axis=1)**2,axis=1)).round(6))
