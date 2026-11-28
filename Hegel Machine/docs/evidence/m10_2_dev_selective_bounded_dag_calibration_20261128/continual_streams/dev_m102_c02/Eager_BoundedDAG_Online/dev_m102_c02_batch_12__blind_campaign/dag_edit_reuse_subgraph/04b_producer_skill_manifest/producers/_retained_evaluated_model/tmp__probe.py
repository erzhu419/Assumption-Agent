import numpy as np,pickle
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def res(a,b):return b-a@np.linalg.lstsq(a,b,rcond=None)[0]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
a,b,c=[S['panel_'+k] for k in 'abc'];u,v=a['u'],a['v'];i,j=np.triu_indices(len(u),1); dx=u[j]-u[i]; g=np.median(np.where(np.abs(dx)>.2,(v[j]-v[i])/dx,np.nan),axis=0);g=np.nanmedian(np.where(np.abs(dx)>.2,(v[j]-v[i])/dx,np.nan),axis=0); o=np.median(v-u*g,0);keep=np.abs(v-o-u*g)<.1
for k in range(4):o[k],g[k]=ls(aff(u[keep[:,k],k,None]),v[keep[:,k],k]);print('A coord',k, o[k],g[k],keep[:,k].sum(),np.sort(np.abs(v[:,k]-o[k]-g[k]*u[:,k])))
for p in [b,c]:
 u=p['u'];print('panel U sv',np.linalg.svd(u,compute_uv=False));
 for k,v in p.items():
  if k=='u':continue
  print(k,'aff',ls(aff(u),v),'sv',np.linalg.svd(v,compute_uv=False)); print('resid first',res(aff(u),v)[:8]);print('loo poly',[(lambda h: np.sqrt(np.mean((res(h,v)/np.maximum(1-np.diag(h@np.linalg.pinv(h)),1e-10)[:,None])**2)))(poly(u,d)) for d in (1,2)])
  print('U,V rows',np.c_[u,v][:10]);print('correlation',np.corrcoef(u.T,v.T)[:4,4:])
T=S['train'];x=(T['x']-o)/g
for d in range(1,5):
 h=poly(x,d); print('train degree',d,'sv beta',np.linalg.svd(ls(h,T['traj']),compute_uv=False),'loo mae y, traj',*[np.mean(np.abs(res(h,np.asarray(y).reshape(len(x),-1))/np.maximum(1-np.diag(h@np.linalg.pinv(h)),1e-10)[:,None])) for y in [T['y'],T['traj']]])
print('Y affine',ls(aff(x),T['y']),'Y quadratic',ls(poly(x,2),T['y']));print('T affine',ls(aff(x),T['traj']));print('train summary', T['summary'],'offset gain',o,g)
