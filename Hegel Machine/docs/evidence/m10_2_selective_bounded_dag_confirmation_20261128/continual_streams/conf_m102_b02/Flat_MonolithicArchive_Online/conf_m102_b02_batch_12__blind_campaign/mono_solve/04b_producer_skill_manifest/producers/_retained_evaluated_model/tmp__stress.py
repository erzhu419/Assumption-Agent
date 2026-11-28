import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; y=np.c_[t['y'],t['traj']]; xx=np.r_[t['x'],s['test']['x']]
def split(i,j):return dict(s,train={k:v[i] if k!='summary' else v for k,v in t.items()},test={'x':t['x'][j]})
ns=np.arange(72); order=np.argsort(np.linalg.norm(t['x'],axis=1)); slices=[('52/20', ns[:52],ns[52:]),('high norm',order[:52],order[52:])]+[(f'block {k}',np.delete(ns,np.arange(k,k+18)),ns[k:k+18]) for k in range(0,72,18)]+[(f'extremes x{k}',(j:=np.argsort(t['x'][:,k]))[9:-9],np.r_[j[:9],j[-9:]]) for k in range(4)]
if __name__=='__main__':
 for name,i,j in slices:
  p=m.solve_batch(split(i,j));e=np.abs(np.c_[p['y_hat'],p['state_pred']]-y[j]); print(name,np.round([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max(),e[-1].max()],6))
 z,h=m._coords(s,xx);d=m._features(z,h); c=m._ls(np.c_[np.ones(72),d[:72]],y); print('scalar coef',np.round(c[:,0],3));print('state coef endpoints',np.round(c[:,[1,8]],3)); print('latent ranges',z.min(0),z.max(0));print('all in fit',np.mean(np.abs(np.c_[np.ones(72),d[:72]]@c-y),0))
