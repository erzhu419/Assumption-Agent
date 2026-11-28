import importlib.util,pickle,numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=np.c_[tr['y'],tr['traj']]; w=m._panels(s,m._coordinates(s,x)); rng=np.random.default_rng(302)
ixs=[np.arange(72),*[rng.permutation(72) for _ in range(16)],np.argsort(np.linalg.norm(x,axis=1)),np.argsort(np.linalg.norm(w,axis=1))]; stats=[]
for ix in ixs:
 i,j=ix[:52],ix[52:]; out=m.solve_batch({**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}); r=np.c_[out['y_hat'],out['state_pred']]-y[j];stats.append([abs(r[:,0]).mean(),abs(r[:,1:]).mean(),abs(r[:,1:]).mean(0).max()])
print('primary, worst random, largest-norm tails',stats[0],np.max(stats[:17],0),stats[-2:]);
for k in ['diag','cross','full']:
 a,b=w[:,:4],w[:,4:]; f=m._bias(np.c_[a,b,np.sum(a*b,axis=1),*([a[:,0]*b[:,1]] if k!='diag' else []),*([m._quadratic(a)[:,5:]] if k=='full' else [])]); co=m._ls(f[:52],y[:52]); r=f@co-y; print(k,'fit',np.std(r[:52],0),'val',abs(r[52:]).mean(0)); print('coef',np.round(co[:,[0,1,-1]],4))
