import pickle,runpy,numpy as np
from itertools import combinations,combinations_with_replacement
D='/holdout_task/'
s=pickle.load(open(D+'public_batch.pkl','rb'))[0]['study']; d=runpy.run_path(D+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); z=d['z']; y=d['y']; T=d['traj']; n=len(y)

def blocks(X):
 a,b,c,e=X.T
 p=np.column_stack([X[:,i]*X[:,j] for i,j in combinations_with_replacement(range(4),2)])
 q=np.column_stack([X[:,i]*X[:,j]*X[:,k] for i,j,k in combinations_with_replacement(range(4),3)])
 h=np.column_stack([np.sin(X),np.cos(X),np.tanh(X),np.arctan(X),X/(1+X*X)])
 ratios=np.column_stack([X[:,i]/(1+X[:,j]**2) for i in range(4) for j in range(4)])
 pairs=np.column_stack([f(X[:,i]+sg*X[:,j]) for f in (np.sin,np.cos,np.tanh,np.arctan) for i,j in combinations(range(4),2) for sg in (-1,1)])
 return [X,np.c_[X,p],np.c_[X,p,q],np.c_[X,p,h],np.c_[X,p,h,ratios,pairs]]

def splits(n):
 ids=np.arange(n); rng=np.random.default_rng(92)
 for vv in [ids[int(n*.7):],*np.array_split(rng.permutation(n),5)]: yield np.setdiff1d(ids,vv),vv

def predict(F,Y,i,j,shrink):
 A,B=F[i],F[j]; mu=A.mean(0); st=np.maximum(A.std(0),1e-9); A=(A-mu)/st; B=(B-mu)/st; m=Y[i].mean(0); U,S,V=np.linalg.svd(A,full_matrices=False); return m+((B@V.T)*(S/(S*S+shrink)))@(U.T@(Y[i]-m))
if __name__=='__main__':
 for name,Y in [('scalar',y),('trajectory',T)]:
  r=[]
  for k,Z in z.items():
   for b,F in enumerate(blocks(Z)):
    for a in ([1e-8,.01,.25,2] if b>1 else [1e-8,.1,1]):
     err=np.array([np.mean(abs(predict(F,Y,i,j,a)-Y[j])) for i,j in splits(n)])
     r.append((err[1:].mean(),err[0],err.max(),k,b,a))
  print(name,'by random CV'); print(*[tuple(round(v,5) if isinstance(v,float) else v for v in q) for q in sorted(r)[:14]],sep='\n'); print(name,'by ordered tail'); print(*[tuple(round(v,5) if isinstance(v,float) else v for v in q) for q in sorted(r,key=lambda q:q[1])[:10]],sep='\n')
