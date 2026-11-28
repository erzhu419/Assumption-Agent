import numpy as np,pickle
from itertools import combinations_with_replacement
np.set_printoptions(precision=4,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; print('times telemetry summary',s['eval_times'],s['telemetry_vec'],tr['summary'])
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def cal(a,x):
 W=[]
 for u,v in zip(a['u'].T,a['v'].T):
  A=aff(u); q=ls(A,v)
  for _ in range(8):
   i=np.argsort(abs(A@q-v))[:42]; q=ls(A[i],v[i])
  keep=abs(A@q-v)<.002;q=ls(A[keep],v[keep]);W.append(q)
 print('calibration',np.array(W));return (x-np.array(W)[:,0])/np.array(W)[:,1]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ii],1) for k in range(1,d+1) for ii in combinations_with_replacement(range(x.shape[1]),k)])
z=cal(s['panel_a'],np.r_[tr['x'],s['test']['x']]); b=s['panel_b']; wb=ls(aff(b['u']),b['v']); w=aff(z)@wb
c=s['panel_c']; u=c['u']; print('panel C rows:',np.c_[u,c['v1'],c['v2']]);
for name in ['v1','v2']:
 Y=c[name]; print(name,'centered SVD',np.linalg.svd(Y-Y.mean(0),compute_uv=False),'unctr',np.linalg.svd(Y,compute_uv=False));
 for d in range(1,4):
  X=poly(u,d); fit=X@ls(X,Y); loo=(fit-Y)/np.maximum(1-np.sum(X*np.linalg.pinv(X).T,axis=1),1e-10)[:,None];print(d,np.mean(abs(fit-Y)),np.mean(abs(loo)))
Y=np.c_[tr['y'],tr['traj']]; print('traj right SV',np.linalg.svd(tr['traj'],full_matrices=False)[2][:3]);print('y corr traj',np.corrcoef(Y.T)[0]);
for name,x in [('raw',np.r_[tr['x'],s['test']['x']]),('z',z),('w',w)]:
 for d in [1,2,3]:
  F=poly(x,d); R=Y[52:]-F[52:72]@ls(F[:52],Y[:52]); print(name,d,np.mean(abs(R[:,0])),np.mean(abs(R[:,1:])),np.mean(abs(R[:,1:]),axis=0))
 for f in [np.tanh,np.arcsinh,np.arctan,np.sin]:
  best=[]
  for g in [.5,1,1.5,2,3,4,6]:
   h=f(g*x); Q=poly(h,2); F=np.c_[aff(x),Q[:,1:]];R=Y[52:]-F[52:72]@ls(F[:52],Y[:52]);best.append((np.mean(abs(R[:,0])),np.mean(abs(R[:,1:])),g))
  print(name,f.__name__, min(best),min(best,key=lambda r:r[1]))
