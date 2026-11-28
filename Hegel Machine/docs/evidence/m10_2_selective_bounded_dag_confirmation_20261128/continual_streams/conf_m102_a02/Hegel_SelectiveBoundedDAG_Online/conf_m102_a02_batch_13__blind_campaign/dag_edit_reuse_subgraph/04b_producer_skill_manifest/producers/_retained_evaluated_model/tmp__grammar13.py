exec(open('/tmp/read13.py').read().split('from itertools import combinations_with_replacement as cr')[0])
from itertools import combinations as cmb, combinations_with_replacement as cr
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Lasso
Y=tr['y']; H=np.c_[np.ones(len(q)),b]; N=len(Y); G=np.cos(p*q)
F=[np.arange(j,min(j+18,N)) for j in (0,18,36,54)]+[np.arange(52,72)]
def cv(A):
 return np.round([np.mean(abs(A[j]@ls(A[i],Y[i])-Y[j])) for j in F for i in [np.setdiff1d(np.arange(N),j)]],5)
def rm(A):return np.sqrt(np.mean((A@ls(A,Y)-Y)**2))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def poly(a,n):return np.column_stack([np.prod(a[:,t],axis=1) for t in cr(range(a.shape[1]),n)])
for nm,a in [('b',b),('g',G),('q',q)]:
 A=np.c_[H,poly(a,2)];print(nm,'quadratic',rm(A),cv(A),list(zip(cr(range(4),2),np.round(ls(A,Y)[5:],4))));
 for r in [3,4]:
  A=np.c_[A,poly(a,r)]; print(nm,r,rm(A),cv(A))
for nm,a in [('B',b),('G',G),('q',q)]:
 for i,j in cmb(range(4),2):
  for c in [-1,1]:
   x=a[:,i]+c*a[:,j]
   for vname,v in [('cos',np.cos(p*x)),('sin',np.sin(p*x)),('tanh',np.tanh(x)),('sq',x*x),('product',a[:,i]*a[:,j])]:
    A=np.c_[H,v]; print('ONE',nm,i,j,c,vname,rm(A),cv(A)) if rm(A)<.085 else None
U=[('H',H),('cyc',np.c_[H,b*np.roll(q,-1,axis=1)]),('quad',np.c_[H,poly(G,2)]),('oddt',np.c_[H,q,b*np.roll(q,-1,axis=1)])]
Z=[]
for nm,V in [('q',q),('x',tr['x']),('b',b),('G',G),('q2',q*q)]:
 Z += [(nm+str(i),V[:,i]) for i in range(4)]; Z += [(f'{nm}{i}{op}{j}',V[:,i]+sgn*V[:,j]) for i,j in cmb(range(4),2) for op,sgn in [('+',1),('-',-1)]];Z += [(f'{nm}{i}*{j}',V[:,i]*V[:,j]) for i,j in cmb(range(4),2)]
Z += [('state',tr['traj'][:,-1]),('statefit',(H@ls(H,tr['traj']))[:,-1])]
results=[]
for nm,z in Z:
 for mode in ('full','offset','partial','raw'):
  for bn,B in U:
   def design(k):
    g=1/(1+k*z*z)
    if mode=='full':return B*g[:,None]
    if mode=='offset':return np.c_[np.ones(N),B*g[:,None]]
    if mode=='partial':return np.c_[B,H*g[:,None]]
    return np.c_[B,g]
   k=minimize_scalar(lambda k:rm(design(k)),bounds=(.001,4),method='bounded').x; A=design(k); rr=rm(A)
   if rr<.059:results.append((rr,nm,mode,bn,round(k,4),cv(A)))
for r in sorted(results,key=lambda r:r[0])[:30]: print('GATE',r)
# Evaluate full pair-harmonic bases and sparse approximation to the residual.
for nm,a in [('g',G),('q',q)]:
 S=np.sin(p*q); M=np.c_[H,poly(G,2),cross(S,S),cross(S,G),S];
 if nm=='q':M=np.c_[M,cross(q,S),cross(q,G),cross(q*q,G),poly(q,2),q,q**3]
 for alpha in (.001,.003,.006,.012):
  out=[]
  for j in F:
   i=np.setdiff1d(np.arange(N),j); d=np.maximum(M[i,1:].std(0),.03); mdl=Lasso(alpha=alpha,max_iter=10000).fit(M[i,1:]/d,Y[i]);out.append(np.mean(abs(mdl.predict(M[j,1:]/d)-Y[j])))
  print('LASSO',nm,alpha,np.round(out,5))
