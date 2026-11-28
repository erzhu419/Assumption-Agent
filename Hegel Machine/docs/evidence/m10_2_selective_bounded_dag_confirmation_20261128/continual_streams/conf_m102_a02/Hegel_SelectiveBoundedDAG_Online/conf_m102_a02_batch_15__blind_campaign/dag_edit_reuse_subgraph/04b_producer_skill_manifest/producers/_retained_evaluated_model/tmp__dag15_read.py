import numpy as np
from itertools import combinations_with_replacement
from sklearn.linear_model import Lasso,OrthogonalMatchingPursuit

def ls(f,y):return np.linalg.lstsq(f,y,rcond=None)[0]
def dictionary(q,b):
 c=np.cos(2.6*q); s=np.sin(2.6*q); n=len(q)
 blocks={}
 def add(name,x,names):blocks[name]=(np.asarray(x).reshape(n,-1),names)
 add('q',q,[f'q{i}' for i in range(4)]); add('b',b,[f'b{i}' for i in range(4)])
 add('cyclic',b*np.roll(q,-1,axis=1),[f'b{i}q{(i+1)%4}' for i in range(4)])
 for name,x,labels in [('Q',q,list('abcd')),('B',b,list('ABCD')),('M',np.c_[q,b],list('abcdABCD')),('C',np.c_[q,c],list('abcdEFGH'))]:
  for d in (2,3):
   ts=list(combinations_with_replacement(range(x.shape[1]),d)); add(name+str(d),np.array([np.prod(x[:,t],1) for t in ts]).T,[''.join(labels[i] for i in t) for t in ts])
 add('sin',s,[f's{i}' for i in range(4)])
 g=[1/(1+k*q[:,i]**2) for k in (.22,1,2.15) for i in range(4)]
 add('rat',np.column_stack([x*v for v in g for x in np.c_[q,b].T]),[f'rat{i}' for i in range(96)])
 return blocks

def des(blocks,keys):return np.column_stack([blocks[k][0] for k in keys]),sum([blocks[k][1] for k in keys],[])

def opts(f,y,t,v,method,p):
 mu=f[t].mean(0); sd=np.maximum(f[t].std(0),1e-6); a=(f[t]-mu)/sd; x=(f[v]-mu)/sd
 ym=y[t].mean(0); yy=y[t]-ym
 if method=='ols':return x@ls(a,yy)+ym
 if method=='ridge':return x@np.linalg.solve(a.T@a+p*np.eye(a.shape[1]),a.T@yy)+ym
 if method=='lasso':
  if yy.ndim>1:return np.column_stack([Lasso(alpha=p, max_iter=7000,tol=1e-5).fit(a,yy[:,j]).predict(x) for j in range(yy.shape[1])])+ym
  return Lasso(alpha=p,max_iter=7000,tol=1e-5).fit(a,yy).predict(x)+ym
 if method=='omp':
  sel=[]; resid=yy
  for i in range(p):
   score=np.linalg.norm((a.T@resid).reshape(a.shape[1],-1),axis=1);score[sel]=-1;sel.append(np.argmax(score)); w=ls(a[:,sel],yy);resid=yy-a[:,sel]@w
  return x[:,sel]@w+ym

if __name__=='__main__':
 for mode in ['a','sine2']:
  d=np.load('/tmp/dag15_'+mode+'.npz'); y=d['y'];z=d['z'];n=len(y); q=d['q'][:n];b=d['b'][:n]; packs=dictionary(q,b); folds=np.array_split(np.arange(n),4)+[np.arange(n-20,n)]; folds += [np.argsort(np.max(abs(q),1))[-16:]]
  for target in [y,z]:
   results=[]
   for keys in [('q',),('b',),('q','b'),('b','cyclic'),('q','b','cyclic'),('q','Q2'),('b','B2'),('q','b','Q2','B2','cyclic'),('q','b','M2'),('q','b','C2'),('q','b','C2','C3'),('q','b','M2','sin'),('q','b','M2','rat')]:
    f,names=des(packs,keys)
    for method,ps in [('ols',[0] if f.shape[1]<40 else []),('ridge',[.03,.3,2,8]),('lasso',[.001,.003,.01] if target.ndim==1 else []),('omp',[4,6,8,12,16])]:
     for p in ps:
      sc=[np.mean(abs(opts(f,target,np.setdiff1d(np.arange(n),v),v,method,p)-target[v])) for v in folds]
      results.append((max(sc[:5]),sc[-2],sc[-1],keys,method,p))
   print(mode,target.shape,'worst ordinary', sorted(results)[:8], 'best official', sorted(results,key=lambda a:a[1])[:5],flush=True)
