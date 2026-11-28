import numpy as np,pickle,itertools as it
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]
np.set_printoptions(precision=5,suppress=True,linewidth=145)
def fit(A,B): return lstsq(A,B,cond=1e-11)[0]
def pol(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in it.combinations_with_replacement(range(x.shape[1]),k)])
def cv(A,B,k=52):
 e=np.abs(A[k:]@fit(A[:k],B[:k])-B[k:]); return [e[:,0].mean(),e[:,1:].mean()]
print('times',s['eval_times'],'tele',s['telemetry_vec'],'traj sv',np.linalg.svd(tr['traj'],compute_uv=False)); print('row sample',np.c_[x,Y][:5]); print('raw CV',[(d,cv(pol(x,d),Y)) for d in range(1,5)])
a=s['panel_a']; u=a['u']; v=a['v']; mask=np.linalg.norm(v-u,axis=1)<2
for name,A in [('aff',pol(u,1)),('quad',pol(u,2)),('diag3',np.c_[pol(u,1),u**2,u**3]),('cub',pol(u,3))]:
 b=fit(A[mask],v[mask]); print('A',name,'rms',np.sqrt(np.mean((A[mask]@b-v[mask])**2,axis=0))); 
 if name=='diag3':print(b)
for n in 'bc':
 p=s['panel_'+n]; u=p['u'];
 for key in list(p)[1:]:
  V=p[key]; print(n,key)
  for name,A in [('linear',pol(u,1)),('diag2',np.c_[pol(u,1),u**2]),('diag3',np.c_[pol(u,1),u**2,u**3]),('pairs',np.c_[pol(u,1),*[u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]]),('quad',pol(u,2)),('quad_diag3',np.c_[pol(u,2),u**3]),('sin1',np.c_[pol(u,1),np.sin(u),np.cos(u)]),('sin2',np.c_[pol(u,1),np.sin(2*u),np.cos(2*u)])]:
   b=fit(A,V); e=A@b-V; h=(np.linalg.pinv(A).T*A).sum(1); print(name, np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2)))
  A=pol(u,2); print('quad coefs',fit(A,V).T)
