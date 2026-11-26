import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w04_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w04_batch_06.pkl','rb'))[0]['dataset']; s=d['train']; dt,t,q=(d[k] for k in ('past_dt','eval_times','yield_stress')); x,_=m._align(s,d['test']); y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; g=m._calibrate(d['calibration_load_cells']); print('affine',g)
labs=s['lab_id']; miss=~np.isfinite(s['coupon_features_raw']).all(1); pred=np.empty((4,*y.shape)); rate=np.empty((4,len(r))); rng=np.random.default_rng(26)
fold=np.empty(len(r),int)
for lab in range(4):
 ii=rng.permutation(np.flatnonzero(labs==lab)); fold[ii]=np.arange(len(ii))%5
for f in range(5):
 a=fold!=f; b=~a; tr={k:v[a] for k,v in s.items()}; te={k:v[b] for k,v in s.items()}
 ts,w=m._fit(dt,t,*m._stress(tr,g),x[a],y[a],r[a],q); print('fold',f,'taus',ts.round(6),'weights',w.round(6))
 ac,ar=m._design(ts,dt,t,*m._stress(tr,g),x[a],q); bc,br=m._design(ts,dt,t,*m._stress(te,g),x[b],q)
 ep=y[a]-ac@w; er=r[a]-ar@w; ir=r[b]-br@w
 dm=m._missing_drift(tr,te,x[a],x[b],w[3:7],ar@w,br@w)
 da=m._alignment_drift(tr,te,x[a],x[b],ep,er,ir,t)
 for j,c in enumerate([np.zeros(b.sum()),dm,da,dm+da]): pred[j,b]=bc@w+c[:,None]*t; rate[j,b]=br@w+c
for name,b in [('all',np.ones(len(r),bool)),('missing',miss),('complete',~miss)]+[(f'lab{i}',labs==i) for i in range(4)]:
 e=pred[:,b]-y[b]; z=rate[:,b]-r[b]; print(name,b.sum(),'strain MAE',np.abs(e).mean((1,2)).round(6),'late MAE',np.abs(e[:,:,-1]).mean(1).round(6),'strain RMSE',np.sqrt((e**2).mean((1,2))).round(6),'rate MAE',np.abs(z).mean(1).round(6),'rate RMSE',np.sqrt((z**2).mean(1)).round(6))
