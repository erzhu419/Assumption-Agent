from p import *
from scipy.linalg import solve
from scipy.optimize import minimize
from itertools import product
edges=[(0,1),(1,2),(2,3),(3,0)]
base=[(),*( (i,) for i in range(4)),*((i,i) for i in range(4)),*edges]
third=sorted({tuple(sorted((*e,i))) for e,i in product(edges,range(4))})
fourth=[tuple(sorted((*e,i,i))) for e,i in product(edges,range(4))]
full2=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]

def features(a,terms):return np.column_stack([np.prod(a[:,k],axis=1) for k in terms])

def fit(F,Y,Fp,groups,opt=True):
 sc=np.sqrt(np.mean(F**2,axis=0)); F=F/sc; Fp=Fp/sc
 size=len(F); I=np.eye(size); mats=[F[:,g]@F[:,g].T for g in groups]; cross=[Fp[:,g]@F[:,g].T for g in groups]
 def objective(h):
  w=np.exp(h); M=sum(j*k for j,k in zip(w,mats))+I; Q=solve(M,I,assume_a='pos'); a=Q@Y; D=Q.diagonal()[:,None]; E=a/D
  J=[]
  for v,K in zip(w,mats):
   qk=Q@K; d=-(qk@a)/D+E*(np.sum(qk*Q,axis=1)[:,None]/D); J.append(2*v*np.mean(d*E))
  return np.mean(E*E),np.array(J)
 h=minimize(objective,np.full(len(groups),4.),jac=True,method='L-BFGS-B',bounds=[(-5,12)]*len(groups),options={'ftol':1e-11}).x if opt else np.full(len(groups),16.)
 w=np.exp(h); a=solve(sum(j*k for j,k in zip(w,mats))+I,Y,assume_a='pos'); C=np.concatenate([]) if False else (F.T@a)/sc[:,None];
 for g,v in zip(groups,w): C[g]*=v
 return sum(j*k for j,k in zip(w,cross))@a,C,h

if __name__=='__main__':
 folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
 for terms,ids in [(base+third+fourth,[1,5,9,13,25]),(full2+third+fourth,[1,5,15,27]),(full2+third+fourth,[1,5,15,27,31,35,39]),(base+third,[1,5,9,13])]:
  A=features(x,terms); G=np.split(np.arange(len(terms)),ids); errs=[]
  for jj in folds:
   ii=np.setdiff1d(np.arange(len(x)),jj); pred=[]
   for sl in (slice(0,1),slice(1,None)):
    yh,_,h=fit(A[ii],Y[ii,sl],A[jj],G);pred.append(yh)
   errs.append([np.mean(abs(pred[0]-Y[jj,:1])),np.mean(abs(pred[1]-T[jj]))]);
  print('groups',ids,'folds',np.round(errs,5));
  if len(terms)==41:
   _,C,h=fit(A,Y,A,G); print('coefficients y, first, last',list(zip(terms,np.round(C[:,[0,1,-1]],3).tolist())))
