import importlib.util, pickle, numpy as np
p=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; idx=np.arange(len(tr['x'])); rng=np.random.default_rng(771)
def errors(fn,i,j):
 v={**s,'train':{k:w[i] if k!='summary' else w for k,w in tr.items()},'test':{'x':tr['x'][j]}}; p=fn(v); return np.abs(p['y_hat']-tr['y'][j]), np.abs(p['state_pred']-tr['traj'][j])
def check(fn):
 for n in (52,56):
  y,a=errors(fn,idx[:n],idx[n:]);print(n, y.mean(),a.mean(),a.mean(0).max())
 vals=[]
 for _ in range(12):
  i=rng.permutation(idx); y,a=errors(fn,i[:52],i[52:]); vals.append([y.mean(),a.mean(),a.mean(0).max()])
 print('random 52/20 median/max',np.median(vals,0),np.max(vals,0))
 z=m._calibrate(s['panel_a'],tr['x']); radius=np.linalg.norm(z,axis=1);i=idx[np.argsort(radius)];y,a=errors(fn,i[:52],i[52:]);print('high radius',y.mean(),a.mean(),a.mean(0).max())
if __name__=='__main__':check(m.solve_batch)
