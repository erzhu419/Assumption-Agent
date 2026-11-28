import pickle,runpy,numpy as np
from itertools import combinations_with_replacement as comb
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a,b,c=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%i)['run_node'] for i in range(3)]
x=a({'__input__':s}); d=b({'node_00':x}); t=c({'__input__':s})
X=x['x_train']; C=x['c_train']; D=d['d_train']; Y=t['target'][:,0]; T=t['target'][:,1:]; n=len(Y); Z=np.c_[X,C]
rng=np.random.default_rng(51); splits=[(np.arange(n-20),np.arange(n-20,n))]+[(p[20:],p[:20]) for p in [rng.permutation(n) for _ in range(8)]]
def cv(F,transform=lambda y:y, inverse=lambda y:y):
 out=[]
 for lam in (0,1e-5,1e-4,.001,.01,.1,1):
  scores=[]
  for i,j in splits:
   A=F[i]; v=transform(Y[i]); sd=np.maximum(.03,A.std(0)); sd[0]=1; A=A/sd; B=F[j]/sd
   p=np.linalg.lstsq(A,v,rcond=1e-10)[0] if lam==0 else solve(A.T@A+np.diag(np.r_[0.,np.full(A.shape[1]-1,lam)]),A.T@v,assume_a='pos')
   scores.append(np.mean(abs(inverse(B@p)-Y[j])))
  out.append((round(np.mean(scores),5),round(max(scores),5),round(scores[0],5),lam))
 return min(out)
def poly(z,k):return np.column_stack([np.ones(len(z)),*[np.prod(z[:,j],axis=1) for p in range(1,k+1) for j in comb(range(z.shape[1]),p)]])
np.savez('/tmp/coords.npz',X=X,C=C,D=D,Y=Y,T=T)
print('ranges',np.min(Z,axis=0),np.max(Z,axis=0),'corr',np.round(np.corrcoef(np.c_[Z,Y,T].T)[-9,:],3))
F={'D':D,'x2':np.c_[D,poly(X,2)[:,5:]],'c2':np.c_[D,poly(C,2)[:,5:]],'z2':poly(Z,2),'z3':poly(Z,3),'x3':np.c_[D,poly(X,3)[:,5:]],'c3':np.c_[D,poly(C,3)[:,5:]]}
for k,f in F.items():
 results=[(cv(f),'identity')]
 for h in (1.95,2.02,2.1,2.2,2.5,3.,4.): results.append((cv(f,lambda y:np.arctanh(y/h),lambda p:h*np.tanh(p)),h))
 print(k,sorted(results)[:4])
p=np.linalg.lstsq(D,Y,rcond=None)[0]; r=Y-D@p
print('baseline residual std',r.std(),'coef',np.round(p,3)); print(np.round(np.c_[X,Y,D@p,r][:20],3))
