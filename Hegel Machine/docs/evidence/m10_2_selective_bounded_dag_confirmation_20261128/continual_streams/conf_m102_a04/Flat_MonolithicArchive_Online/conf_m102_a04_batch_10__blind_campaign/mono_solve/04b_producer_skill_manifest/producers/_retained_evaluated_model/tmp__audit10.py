import pickle, sys, importlib.util, numpy as np
spec=importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); z=m._calibrate(s['panel_a'],tr['x']); g,h=m._panel_maps(s['panel_c'],z)
Y=np.c_[tr['y'],tr['traj']]; cross=(g[:,:,None]*h[:,None,:]).reshape(n,16); D=m.one(np.c_[g,h,cross]); D0=m.one(np.c_[g,h,(g*h).sum(1)]);
for F,tar in [(D0,Y[:,0]),(D,Y[:,0]),(D,Y[:,1:])]:
 coef=m.lin(F[:52],tar[:52]); res=F@coef-tar; print('matrix',F.shape, 'MAE train, hold',np.mean(abs(res[:52])),np.mean(abs(res[52:])), 'coef',np.round(coef if coef.ndim==1 else coef[:,-1],5)); print('row errors', np.round(np.sort(np.abs(res) if res.ndim==1 else np.mean(np.abs(res),1))[-5:],5))
for test in [np.arange(52,72),np.argsort(np.linalg.norm(z,axis=1))[-20:], *(a[:20] for a in [np.random.default_rng(i).permutation(n) for i in range(5)])]:
 fit=np.setdiff1d(np.arange(n),test); q=dict(s,train={**tr,**{k:tr[k][fit] for k in ('x','y','traj')}},test={'x':tr['x'][test]}); out=m.solve_batch(q); e=abs(out['y_hat']-Y[test,0]); et=abs(out['state_pred']-Y[test,1:]); print('split, y MAE, trajectory MAE, worst time MAE:',np.round([e.mean(),et.mean(),et.mean(0).max()],6))
