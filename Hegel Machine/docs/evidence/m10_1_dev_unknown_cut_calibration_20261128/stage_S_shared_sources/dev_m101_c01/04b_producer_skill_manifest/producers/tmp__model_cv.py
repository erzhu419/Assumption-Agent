import pickle
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,z]
E=np.c_[-np.expm1(-1.52*t),-np.expm1(-.22*t)]; P=E@np.linalg.pinv(E)
def aff(x):return np.c_[np.ones(len(x)),x]
def terms(x,k):return np.column_stack([np.prod(x[:,q],axis=1) for q in cr(range(4),k)])
F=np.c_[aff(x),terms(x,2)];G=np.c_[F,terms(x,3)];D=x@x.T/4;R=cdist(x,x,'sqeuclidean');I=np.eye(len(x));
b=np.linalg.lstsq(G,Y,rcond=None)[0];p=s['panel_b'];bw=np.linalg.lstsq(aff(p['u']),p['v'],rcond=None)[0]; print('B directions',bw[1:].T.round(3),'orthogonal', (bw[1:].T@bw[1:]).round(3));
for i in [0,1,8]:
 H=np.zeros((4,4))
 for (j,k),v in zip(cr(range(4),2),b[5:15,i]):H[j,k]=H[k,j]=v*(1 if j==k else .5)
 print('quadratic',i,np.round(np.linalg.eigvalsh(H),4),'B basis',np.round(bw[1:].T@H@bw[1:],4));print('linear in B basis',np.round(bw[1:].T@b[1:5,i],4))
splits=list(np.array_split(np.arange(len(x)),6))+[np.argsort(np.linalg.norm(x,axis=1))[-12:]]
def evaluate(pred):
 errs=[]
 for j in splits:
  i=np.setdiff1d(np.arange(len(x)),j);r=abs(pred(i,j)-Y[j]);errs.append([r[:,0].mean(),r[:,1:].mean(),r[:,-1].mean(),max(r[:,1:].mean(0))])
 return np.array(errs)
def output(name,pred,rows):
 e=evaluate(pred);rows.append((name,e[:6].mean(0),e.max(0),e[-1]));
def project(a):
 a[:,1:]=a[:,1:]@P;return a
rows=[]
for d in (2,3,4,5):
 A=np.c_[F,*[terms(x,k) for k in range(3,d+1)]];g=A@A.T
 for l in (1e-6,.001,.01,.1):output(('mono',d,l),lambda i,j:project(g[np.ix_(j,i)]@solve(g[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos')),rows)
for d in (3,4,5,6,8):
 for amp in (.1,.5,1.):
  g=(1+amp*D)**d
  for l in (1e-6,1e-4,.001):output(('poly',d,amp,l),lambda i,j:project(g[np.ix_(j,i)]@solve(g[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos')),rows)
for length in (0.65,1.,1.4,2.,3.,4.):
 for l in (1e-5,.0001,.001,.01):
  for family in ('rbf','m52'):
   q=np.sqrt(5*R)/length;g=np.exp(-R/(2*length**2)) if family=='rbf' else (1+q+q*q/3)*np.exp(-q);g=g+F@F.T
   output((family,length,l),lambda i,j:project(g[np.ix_(j,i)]@solve(g[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos')),rows)
# Rational polynomial models: regression of numerator against y times denominator.
for d in (2,3):
 A=F if d==2 else G
 for Q in (x,terms(x,2),np.c_[x,terms(x,2)],np.sum(x*x,axis=1,keepdims=True)):
  for lr in (.001,.03,.15):
   def pred(i,j):
    out=[]
    for q in Y.T:
     mat=np.c_[A[i],-Q[i]*q[i,None]];pen=np.r_[np.repeat(.0001,A.shape[1]),np.repeat(lr,Q.shape[1])]; w=solve(mat.T@mat+np.diag(pen),mat.T@q[i],assume_a='pos');den=1+Q[j]@w[A.shape[1]:];out.append((A[j]@w[:A.shape[1]])/np.maximum(.45,den))
    return project(np.array(out).T)
   output(('rat',d,Q.shape[1],lr),pred,rows)
for j in [0,1,2,3]:
 print('BEST',j)
 for name,a,b,c in sorted(rows,key=lambda r:r[1][j]+.7*r[2][j])[:9]:print(name,a.round(5),b.round(5),c.round(5))
