from panel6 import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
import importlib.util
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m101_b01_batch_05_solver.py'); old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
x=old._inverse_calibration(tr['x'],old._calibration(a)); u=b['u']; v=b['v']
def bd(x):return af(x**3/np.sqrt(1+.70020846*x**4)-.08899942*x)
B=lambda x:bd(x)@ls(bd(u),v)
def c1d(x):return af(cross(x)[:,[0,2,3,5]])
C1=lambda x:c1d(x)@ls(c1d(c['u']),c['v1'])
def c2d(x):return np.c_[af(x),np.cos(1.28247*x)]
C2=lambda x:c2d(x)@ls(c2d(c['u']),c['v2'])
z=B(x); z0=old._operator_b(b,x); q=C1(z); r=C2(z); bx=bd(x)[:,1:]; print('traj mean',Y[:52,1:].mean(0),'scalar stats',np.quantile(Y[:52,0],[0,.1,.5,.9,1])); print('rank modes',np.linalg.svd(Y[:52]-Y[:52].mean(0),full_matrices=False)[2][:2]); print('corr y/state',np.corrcoef(Y[:52].T)[0]);print('summary',s['train']['summary'])
folds=[(np.setdiff1d(np.arange(52),i),i) for i in np.array_split(np.random.default_rng(624).permutation(52),4)]; final=(np.arange(52),np.arange(52,72));
def st(x):return x/np.maximum(np.sqrt(np.mean(x[:52]**2,0)),1e-8)
def gram(x):return x@x.T/max(1,x.shape[1])
def unary(x):
 x=st(x);return np.c_[np.tanh(x), np.tanh(2*x), np.sin(x),np.cos(x),x/(1+x*x),1/(1+x*x),np.abs(x),x*np.abs(x), x*x, x**3, x/(1+.25*x*x),x*x/(1+x*x)]
def pair(x):
 x=st(x); i,j=np.triu_indices(x.shape[1],1);return np.c_[x[:,i]*x[:,j], (x[:,i]+x[:,j])*.7071, (x[:,i]-x[:,j])*.7071, x[:,i]/(1+x[:,j]**2),x[:,j]/(1+x[:,i]**2)]
def examine(G):
 def fit(i,j,n):
  m=Y[i].mean(0);return abs(G[np.ix_(j,i)]@solve(G[np.ix_(i,i)]+n*np.eye(len(i)),Y[i]-m,assume_a='pos')+m-Y[j]);
 for n in [1e-5,.00015,.002,.02]:
  E=np.concatenate([fit(i,j,n) for i,j in folds]); e=fit(*final,n); yield n,[E[:,0].mean(),E[:,1:].mean(),e[:,0].mean(),e[:,1:].mean()]
res=[]
for name,Q in {'x':x,'B':z,'B0':z0,'C1':q,'C2':r,'B+C1':np.c_[z,q],'B+C2':np.c_[z,r],'B+C':np.c_[z,q,r],'x+B+C':np.c_[x,z,q,r],'cube+B+C':np.c_[bx,z,q,r],'x+cube+B+C':np.c_[x,bx,z,q,r],'C(x)+B+C':np.c_[x,C1(x),C2(x),z,q,r],'BC1(x)+BC2(x)+B+C':np.c_[x,B(C1(x)),B(C2(x)),z,q,r],'Bmean+C':np.c_[z,q,r,(z+z0)*.5]}.items():
 Q=st(Q); G=gram(Q); U=gram(st(unary(Q))); P=gram(st(pair(Q))); D=cdist(Q,Q,'sqeuclidean')/Q.shape[1]; D/=2
 for k,K in {'poly2':(1+G)**2,'poly3':(1+G)**3,'poly4':(1+G)**4,'unary':G+U,'unarypair':G+U+P,'unarypair2':G+U+P+gram(st(unary(pair(Q)))),'poly+unary':(1+G)**2+U+P,'mixed':(1+G+U+P)**2,'rbf':G*.3+np.exp(-D/2.25),'rational':G*.3+1/(1+D/5)}.items():
  for n,e in examine(K):res.append((name,k,n,*e))
for j in [3,4]:
 print('best',j);print(*sorted(res,key=lambda r:r[j])[:12],sep='\n')
