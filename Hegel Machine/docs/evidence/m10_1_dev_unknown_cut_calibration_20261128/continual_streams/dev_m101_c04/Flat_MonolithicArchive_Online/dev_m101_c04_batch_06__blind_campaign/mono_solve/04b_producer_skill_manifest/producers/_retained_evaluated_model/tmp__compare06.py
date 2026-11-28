import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; z,zt=m._coordinates(s)
rng=np.random.default_rng(9); rows=np.arange(len(y)); folds=[np.arange(56,72),*np.array_split(rng.permutation(len(y)),5),np.argsort(np.linalg.norm(z,axis=1))[-12:]]
for mode in ['contrast','full','sparse','symmetric','aligned']:
 vals=[]; horizon=[]
 for j in folds:
  i=np.setdiff1d(rows,j); yp,tp=m._predict(z[i],z[j],y[i],T[i],mode); delta=np.abs(tp-T[j]); vals.append([np.mean(abs(yp-y[j])),delta.mean(),delta[:,-1].mean()]); horizon.append(delta.mean(axis=0))
 print(mode,'y MAEs',np.round(np.array(vals)[:,0],6),'ordinary mean and max',np.round([np.mean(np.array(vals)[:6,0]),np.max(np.array(vals)[:6,0])],6),'trajectory max by horizon',np.round(np.max(horizon,axis=0),6))
