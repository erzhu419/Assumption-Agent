import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['train']; y=p['y']; T=p['traj']; z,c=m._coordinates(s,np.r_[p['x'],s['test']['x']]); z,c=z[:72],c[:72];
folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(64,73)]+[np.argsort(np.linalg.norm(z,axis=1))[-20:]]
for tag in ('elastic','cyclic','tied','tiedextra','cyclicextra'):
 errs=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(72),te); ix=np.r_[tr,te]
  if tag=='elastic': a,b=m._readout(z[ix],c[ix],y[tr],T[tr]); P=np.c_[a,b]
  else:
   q=z*np.roll(c,1,axis=1); extra=c[:,0]*c[:,2]; A=np.c_[np.ones(72),c,q if tag.startswith('cyclic') else q.sum(1)];
   U=np.c_[A,extra] if tag.endswith('extra') else A
   P=np.c_[U[te]@m._fit(U[tr],y[tr]), A[te]@m._fit(A[tr],T[tr])]
  E=abs(P-np.c_[y[te],T[te]]);errs.append((E[:,0].mean(),E[:,1:].mean(),E[:,-1].mean()))
 e=np.array(errs); print(tag,'gate',e[0].round(5),'other mean/max',e[1:-1].mean(0).round(5),e[1:-1].max(0).round(5),'outer-radius',e[-1].round(5))
