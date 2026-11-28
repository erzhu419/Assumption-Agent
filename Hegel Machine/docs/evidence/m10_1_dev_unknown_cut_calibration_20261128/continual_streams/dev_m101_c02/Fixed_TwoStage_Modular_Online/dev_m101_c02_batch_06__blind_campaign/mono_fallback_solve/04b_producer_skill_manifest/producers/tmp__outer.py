import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_models
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,z=_coordinates(s,tr['x']); y=np.c_[tr['y'],tr['traj']]; n=len(y)
splits=[np.arange(n)]+[np.random.default_rng(k).permutation(n) for k in range(10)]+[np.argsort(np.linalg.norm(q/q.std(0),axis=1)) for q in (x,z)]
methods=['joint','pair','mix','weighted','select','ensemble']; results={m:[] for m in methods}
for ii in splits:
 a,b=ii[:56],ii[56:]; mods=list(_models(x[ii],z[ii],y[a]));
 for m in methods:
  ms=[d for d in mods if d[0]==m] or mods; score=np.array([d[1] for d in ms]); p=np.array([d[2] for d in ms]); obj=score@[1,1.4,.2]; w=np.exp(-(obj-obj.min())/.013); yp=p[obj.argmin()] if m=='select' else np.einsum('m,mij->ij',w/w.sum(),p); e=abs(yp-y[b]); results[m].append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
for m,q in results.items():
 e=np.array(q); print(m,'56/16, random mean, random worst, radial slices',np.round(np.r_[e[:1],e[:11].mean(0)[None],e[:11].max(0)[None],e[11:]],5).tolist())
