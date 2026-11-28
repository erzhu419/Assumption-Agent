import runpy,pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w']; y=s['train']['y']; tr=s['train']['traj']; p=s['panel_c']; u=p['u']; n=len(y)
fun={'lin':lambda z:z,'atan':np.arctan,'asinh':np.arcsinh,'tanh':np.tanh,'sin':np.sin,'inv3':lambda z:np.sinh(np.arcsinh(z)/3),'rat':lambda z:z/(1+z*z),'soft':lambda z:z/np.sqrt(1+z*z),'bend':lambda z:z-z/np.sqrt(1+z*z),'exp':lambda z:-np.sign(z)*np.expm1(-abs(z))}
def pair(z,diag=True):return np.column_stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2) if diag or i!=j])
def poly(z,d):return np.column_stack([np.prod(z[:,k],axis=1) for a in range(1,d+1) for k in itertools.combinations_with_replacement(range(z.shape[1]),a)])
def basis(x,k,g,m):
 z=fun[k](g*x); o=np.ones(len(x)); P=pair(z); A=[z,np.c_[z,P],np.c_[x,z,P],np.c_[x,z,P,pair(x)],poly(z,3),np.c_[x,z,pair(z,False)],np.c_[x,z,z*z],np.c_[z,pair(z,False)]][m];return np.c_[o,A]
def reg(A,b,r):
 z=A.std(0);z[z<1e-12]=1; B=A/z
 if r==0:return np.linalg.lstsq(A,b,rcond=1e-9)[0]
 return np.linalg.solve(B.T@B+np.diag(np.r_[0,np.full(A.shape[1]-1,r)]),B.T@b)/z.reshape((-1,)+(1,)*(b.ndim-1))
def check(x,b,folds):
 out=[]
 for k,m,g,r in itertools.product(fun,range(8),[.6,1.2,2.4,4.8,8.], [0,.03,.35]):
  if k=='lin' and g!=1.2:continue
  A=basis(x,k,g,m); ae=[]; train=[]
  for a,v in folds:
   q=reg(A[a],b[a],r); ae.append(np.mean(abs(A[v]@q-b[v]))); train.append(np.mean(abs(A[a]@q-b[a])))
  out.append((np.mean(ae),max(ae),ae[0],np.mean(train),k,m,g,r))
 for line in sorted(out)[:14]:print(tuple(round(f,5) if isinstance(f,(float,np.floating)) else f for f in line))
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(np.random.default_rng(9).permutation(n),4)]
for name,b in [('scalar',y),('traj',tr)]:print(name);check(w[:n],b,folds)
print('panel relationships')
for key,x in [('u',u),('v1',p['v1'])]:
 for ky in ['v1','v2']:
  if key==ky:continue
  b=p[ky];print(key,ky,'corr',np.round(np.corrcoef(x.T,b.T)[:4,4:],2));check(x,b,[(np.arange(18),np.arange(18,24)),(np.arange(6,24),np.arange(6))])
