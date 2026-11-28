import numpy as np, pickle, runpy
root='/holdout_task/'
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
f0,f1,f2=[runpy.run_path(root+p)['run_node'] for p in ('_frozen_nodes/node_00.py','node_01.py','_frozen_nodes/node_02.py')]
def check(j, calibration=None):
 i=np.setdiff1d(np.arange(n),j)
 st=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':x[j]})
 if calibration is not None:
  key,ind=calibration; st[key]={k:v[ind] for k,v in s[key].items()}
 p=f0({'__input__':st}); q=f1({'__input__':st,'node_00':p}); o=f2({'node_00':p,'node_01':q})
 dy=abs(o['y_hat']-tr['y'][j]); dt=abs(o['state_pred']-tr['traj'][j]); return np.r_[dy.mean(),dt.mean(),dt.mean(0)]
folds=[j for seed in (573,309,615) for j in np.array_split(np.random.default_rng(seed).permutation(n),4)]
outer=[np.argsort(k)[-18:] for k in [np.linalg.norm(x-x.mean(0),axis=1),*x.T,*(-x.T)]]
for name,rows in [('random',[check(j) for j in folds]),('coordinate/extreme',[check(j) for j in outer]),('calibration deletion',[check(folds[a%len(folds)],(key,np.delete(np.arange(len(s[key]['u'])),a))) for key in ('panel_a','panel_c') for a in range(len(s[key]['u']))])]:
 m=np.max(rows,axis=0); print(name,'worst y / trajectory / individual horizon',np.round(np.r_[m[:2],m[2:].max()],6)); assert np.max(m) < 0.038
