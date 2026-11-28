import pickle,numpy as np,sys,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; A=f._affine(u); x=f._inverse(s['train']['x'],f._fit_sensor(s['panel_a'])); y=s['train']['y']; tr=s['train']['traj']; U,D,V=np.linalg.svd(tr,full_matrices=False); H=np.c_[y,U[:,:2]*D[:2]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for n in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),n)])
def cv(A,H,ids=None):
 if ids is None:ids=np.arange(len(A))
 i=ids[:-20];j=ids[-20:];return np.round(abs(A[j]@fit(A[i],H[i])-H[j]).mean(0),5)
print('range x',x.min(0),x.max(0),'Y',y.min(),y.max(),'traj V',np.round(V[:2],3)); print('raw polynomial cv',[cv(poly(x,k),H) for k in (1,2,3,4)])
for pk in ['panel_b','panel_c']:
 p=s[pk];u=p['u'];A=f._affine(u)
 for name,v in p.items():
  if name=='u':continue
  print('\n',pk,name,'range',v.min(0),v.max(0));print('norms',np.round(np.linalg.norm(v,axis=1),2))
  for fun in [lambda z:z, np.sin,np.tanh,lambda z:z/(1+z*z),np.sinh,lambda z:z**3]:
   for basis in [A,poly(u,2)]:
    c=fit(basis,v); sol=least_squares(lambda c:(fun(basis@c.reshape(-1,4))-v).ravel(),c.ravel(),max_nfev=100)
    err=np.sqrt(np.mean(sol.fun**2));print(fun.__name__,basis.shape[1],round(err,6),end='; ')
   print()
  for g in [np.sin,np.tanh,np.sinh,lambda z:z**2,lambda z:z**3]:
   res=v-np.c_[A,g(u)]@fit(np.c_[A,g(u)],v);print('input',g.__name__,round(np.sqrt(np.mean(res**2)),6),end='; ')
  print();
  for j in range(4):
   def ff(c, a):return c[0]+c[1]*np.tanh(a@c[2:])
   init=np.r_[v[:,j].mean(),max(abs(v[:,j]))*1.25,fit(A,v[:,j])/(max(abs(v[:,j]))*1.25)]
   sol=least_squares(lambda c:ff(c,A)-v[:,j],init,max_nfev=500);print('offset tanh',j,np.sqrt(np.mean(sol.fun**2)),np.round(sol.x,3));
