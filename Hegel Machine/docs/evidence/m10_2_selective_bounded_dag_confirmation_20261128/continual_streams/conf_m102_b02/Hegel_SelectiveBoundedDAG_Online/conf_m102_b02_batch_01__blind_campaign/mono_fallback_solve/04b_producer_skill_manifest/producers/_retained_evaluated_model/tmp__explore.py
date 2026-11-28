import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=110)
def aff(u,v,rob=False):
 U=np.c_[np.ones(len(u)),u];ix=np.arange(len(u))
 if rob:
  for _ in range(5):
   E=np.linalg.norm(U@np.linalg.lstsq(U[ix],v[ix],rcond=None)[0]-v,axis=1);ix=np.argsort(E)[:44]
 return np.linalg.lstsq(U[ix],v[ix],rcond=None)[0]
def poly(x,n,diag=False):
 return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for j in range(1,n+1) for i in combinations_with_replacement(range(x.shape[1]),j) if not diag or len(set(i))==1])
a=aff(s['panel_a']['u'],s['panel_a']['v'],True)
def cal(x):return np.linalg.solve(a[1:].T,(x-a[0]).T).T
x=cal(s['train']['x']); Y=np.c_[s['train']['y'],s['train']['traj']];print('t',s['eval_times'],'a',a,'x range',x.min(0),x.max(0)); print('signals',Y[:10,[0,1,-1]])
def fits(u,v):
 for name,P in [('lin',poly(u,1)),('diag2',poly(u,2,True)),('full2',poly(u,2)),('diag3',poly(u,3,True)),('diag4',poly(u,4,True)),('sin',np.c_[poly(u,1),np.sin(2*u),np.cos(2*u)]),('asinh',np.c_[poly(u,1),np.arcsinh(3*u)])]:
  Pi=np.linalg.pinv(P); e=(v-P@Pi@v)/(1-(P*Pi.T).sum(1))[:,None];print(name,np.sqrt((e**2).mean(axis=0)).round(5),end='; ')
 print();W=poly(u,1); L=np.linalg.lstsq(W,v,rcond=None)[0]
 for name,f in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('softsign',lambda z:z/np.sqrt(1+z*z)),('square',np.square),('cube',lambda z:z**3)]:
  for extra in [False,True]:
   def calc(p,xx):
    h=f(p[0]+xx@p[1:5]);return p[5]+p[6]*h+(xx@p[7:] if extra else 0)
   rs=[]
   for j in range(4):
    p0=np.r_[0,L[1:,j],L[0,j],1,np.zeros(4) if extra else []];sol=least_squares(lambda p:calc(p,u)-v[:,j],p0,max_nfev=600);rs.append(np.std(sol.fun))
   print(name,extra,np.round(rs,6),end='; ')
  print()
if __name__=='__main__':
 for k in ['panel_b','panel_c']:
  p=s[k];print(k)
  for j,v in p.items():
   if j!='u':print(j);fits(p['u'],v)
