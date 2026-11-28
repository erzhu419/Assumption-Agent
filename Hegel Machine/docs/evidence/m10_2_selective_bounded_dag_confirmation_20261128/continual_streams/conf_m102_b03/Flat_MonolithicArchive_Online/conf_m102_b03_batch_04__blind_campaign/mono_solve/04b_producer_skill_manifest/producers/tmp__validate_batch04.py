import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; train=s['train']; x=train['x']; y=train['y']; t=train['traj']; a,b=m._features(s,x)
rng=np.random.default_rng(42); norms=np.linalg.norm(x,axis=1); splits=[('official',np.arange(52),np.arange(52,72))]
for j in range(40):
 p=rng.permutation(72); splits.append((f'random-{j}',p[:52],p[52:]))
for j,v in enumerate([norms,*x.T,*(-x.T)]):
 p=np.argsort(v); splits.append((f'extrapolation-{j}',p[:52],p[52:]))
records=[]
for name,i,j in splits:
 q={**s,'train':{**train,**{k:train[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}
 out=m.solve_batch(q); ey=abs(out['y_hat']-y[j]); et=abs(out['state_pred']-t[j]); records.append((name,ey.mean(),et.mean(),et.mean(0).max(),ey.max(),et.max()))
print('official:',records[0]);print('worst across alternate splits by metric:',*[max(records,key=lambda r:r[k]) for k in range(1,6)],sep='\n');assert max(max(r[1:4]) for r in records)<.038
h=m._basis(np.asarray(s['eval_times']),t); amp=m._ls(h,t.T).T
print('endpoint coefficients',m._ls(a,y)); print('mode coefficients',m._ls(b,amp).T)
print('trajectory denoising MAE',abs(amp@h.T-t).mean());u=s['panel_c']['u']; p,q=m._operators(s['panel_c'],u);print('panel fit RMS', np.sqrt(((p-s['panel_c']['v1'])**2).mean()), np.sqrt(((q-s['panel_c']['v2'])**2).mean()))
out=m.solve_batch(s);v=out['state_pred'][:,-1]; np.testing.assert_allclose(out['diagnostics_vec'],[out['y_hat'].mean(), out['y_hat'].std(),v.mean(),v.std()]);print('full test diagnostics',out['diagnostics_vec'])
