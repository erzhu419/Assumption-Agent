import copy, importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x);
def run(i,j,panel=None,k=None):
 st=copy.deepcopy(s); st['train']={a:tr[a][i] for a in ('x','y','traj')}; st['test']={'x':x[j]}
 if panel is not None:
  st[panel]={key:np.delete(val,k,axis=0) for key,val in st[panel].items()}
 o=m.solve_batch(st); y=abs(o['y_hat']-tr['y'][j]); z=abs(o['state_pred']-tr['traj'][j]); return np.r_[y.mean(),z.mean(),z.mean(0)]
split=(np.arange(52),np.arange(52,n)); print('Specified split:',np.round(run(*split),6)); rng=np.random.default_rng(671)
E=[]
for _ in range(30):
 a=rng.permutation(n); E.append(run(a[:52],a[52:]));
print('Random 52/20 worst:',np.round(np.max(E,axis=0),6)); E=[]
for q in [*x.T,np.linalg.norm(x,axis=1)]:
 for direction in (-1,1):
  a=np.argsort(direction*q); E.append(run(a[:52],a[52:]));
print('Coordinate tails worst:',np.round(np.max(E,axis=0),6));
E=[run(*split,p,k) for p in ('panel_a','panel_c') for k in range(len(s[p]['u']))]; print('Single panel-row deletion worst:',np.round(np.max(E,axis=0),6))
