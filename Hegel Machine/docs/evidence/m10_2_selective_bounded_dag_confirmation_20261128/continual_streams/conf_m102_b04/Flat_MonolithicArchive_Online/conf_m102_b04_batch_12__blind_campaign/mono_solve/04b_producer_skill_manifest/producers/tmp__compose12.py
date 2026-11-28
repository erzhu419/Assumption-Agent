from panel12 import *
from scipy.optimize import least_squares
from itertools import product
np.set_printoptions(precision=5,suppress=True,linewidth=180)
a=s['panel_a']; keep=np.linalg.norm(a['v']-a['u'],axis=1)<4; u,v=a['u'][keep],a['v'][keep]; terms=[()]+[i for d in (1,2,3) for i in combinations_with_replacement(range(4),d)]; w=ls(poly(u,3),v)
print('A support',[(i,v) for i,v in zip(terms,w) if np.max(abs(v))>.002]); w[abs(w)<.002]=0
for j in range(4):
 m=w[:,j]!=0; w[m,j]=ls(poly(u,3)[:,m],v[:,j]);
print('A rmse',np.sqrt(np.mean((poly(u,3)@w-v)**2)))
def cal(x):
 return np.array([least_squares(lambda z:poly(z[None,:],3)[0]@w-r,r,max_nfev=100).x for r in x])
b=s['panel_b']; c=s['panel_c']; B=lambda x:aff(np.cos(2.6*x)); C=lambda x:aff(np.c_[x,np.arcsinh(3.48452*x)]); D=lambda x:aff(np.logaddexp(0,2.2*x)); funcs={'b':lambda x:B(x)@ls(B(b['u']),b['v']),'c':lambda x:C(x)@ls(C(c['u']),c['v1']),'d':lambda x:D(x)@ls(D(c['u']),c['v2'])}
x=tr['x']; z=cal(x); print('range z',z.min(0),z.max(0),'cal max inverse err',abs(poly(z,3)@w-x).max());
t=s['eval_times']; traj=tr['traj'];
def tb(r):return -np.expm1(-np.outer(t,r))
r=least_squares(lambda r:(tb(r)@ls(tb(r),traj[:52].T)-traj[:52].T).ravel(),[.22,2.15],bounds=(.001,20)).x; T=tb(r); A=ls(T,traj.T).T; y=tr['y']; print('rates',r,'time rmse',np.sqrt(np.mean((A@T.T-traj)**2))); print('y/A correlation',np.corrcoef(np.c_[y,A].T)); print('y vs A polynomial',[(d,score(poly(A,d),y)) for d in (1,2,3)])
if __name__=='__main__':
 X={'x':x,'z':z};
 for start in ['x','z']:
  for sz in (1,2,3):
   for names in product(funcs,repeat=sz):
    f=X[start];
    for name in names: f=funcs[name](f)
    X[start+''.join(names)]=f
 for tag in ['y','a0','a1','tr']:
  tar={'y':y,'a0':A[:,0],'a1':A[:,1],'tr':traj}[tag]; results=[]
  for k,f in X.items():
   for d in [1,2,3]:
    P=poly(f,d); results.append((score(P,tar),k,d,np.mean(abs(P[:52]@ls(P[:52],tar[:52])-tar[:52]))))
  print(tag,sorted(results)[:16])
