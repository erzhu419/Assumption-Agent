import pickle,itertools,numpy as np
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=160)
def poly(x,d):
 return np.column_stack([np.prod(x[:,i],axis=1) for j in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),j)])
def fit(a,b,n):
 c=lstsq(a[:n],b[:n])[0];e=a[n:]@c-b[n:];return np.mean(abs(e),axis=0)
for key in ('panel_a','panel_b','panel_c'):
 p=s[key];x=p['u'];print(key)
 for k,v in p.items():
  if k=='u':continue
  a=poly(x,1);c=lstsq(a,v)[0];r=np.linalg.norm(a@c-v,axis=1); ix=np.argsort(r)[:int(.8*len(x))];c=lstsq(a[ix],v[ix])[0];r=np.linalg.norm(a@c-v,axis=1);print(k,'affine',c,'resid quantiles',np.quantile(r,[0,.25,.5,.75,.85,.9,1]),'largest',np.argsort(r)[-8:]);print('poly holdout',[(j,fit(poly(x,j),v,int(.8*len(x)))) for j in (1,2)])
x,y,T=(s['train'][k] for k in ('x','y','traj'));t=s['eval_times'];u,d,v=np.linalg.svd(T,full_matrices=False);print('times',t,'SVD',d,'basis',v[:4],sep='\n');print('test x abs quantiles',np.quantile(abs(s['test']['x']),[.5,.9,1],axis=0));
for j in (1,2,3):
 a=poly(x,j);print('poly',j,'y',fit(a,y,56),'traj',fit(a,T,56),'scores',fit(a,u[:,:3]*d[:3],56))
for j in (1,2,3):
 print('y from trajectory scores',j,fit(poly(u[:,:3]*d[:3],j),y,56),'y plus x',fit(poly(np.c_[x,u[:,:2]*d[:2]],j),y,56))
print('summary actual',y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std())
