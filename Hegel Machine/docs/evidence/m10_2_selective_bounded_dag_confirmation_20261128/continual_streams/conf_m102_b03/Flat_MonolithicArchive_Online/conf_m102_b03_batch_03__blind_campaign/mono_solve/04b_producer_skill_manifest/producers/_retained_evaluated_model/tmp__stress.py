import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; x=tr['x']; n=len(x); z=m._calibrate(s['panel_a'],x); p,q=m._features(s,x)
rng=np.random.default_rng(512); order=np.argsort(np.linalg.norm(z,axis=1)); tests=[('required',np.arange(52,n)),('large latent norm',order[-20:]),('small latent norm',order[:20])]; tests += [(f'random-{i}',rng.choice(n,20,False)) for i in range(30)];
for j in range(4):
    order=np.argsort(z[:,j]); tests += [(f'low-{j}',order[:12]),(f'high-{j}',order[-12:])]
results=[]
for name,iv in tests:
    it=np.setdiff1d(np.arange(n),iv); trn={k:tr[k][it] for k in ['x','y','traj']}; out=m.solve_batch(dict(s,train=trn,test={'x':x[iv]})); yerr=np.abs(out['y_hat']-tr['y'][iv]); terr=np.abs(out['state_pred']-tr['traj'][iv]); results.append((name,yerr.mean(),terr.mean(),terr.mean(0).max(),terr[:,-1].mean()));
print('required:',results[0]); print('worst by metric:',*[max(results,key=lambda row:row[j]) for j in range(1,5)]); print('actual test ranges:',[(k,np.shape(v),np.min(v),np.max(v)) for k,v in m.solve_batch(s).items()]);
