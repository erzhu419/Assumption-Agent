import runpy
import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
m=runpy.run_path('/tmp/probe07.py'); s,x,y,z,w,ls,aff=[m[k] for k in ('s','x','y','z','w','ls','aff')]
u=s['panel_b']['u']; v=s['panel_b']['v']
print('B fit residual comparisons')
for mode in ('locked','free','colfreq','quadratic','cyclic','cub'):
 def design(x,p):
  if mode=='locked':return aff(x+p[1]*np.sin(p[0]*x))
  if mode=='colfreq':return np.c_[aff(x),np.sin(p*x)]
  a=np.c_[aff(x),np.sin(p[0]*x)]
  if mode=='quadratic':a=np.c_[a,*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]]
  if mode=='cyclic':a=np.c_[a,x*np.roll(x,-1,axis=1)]
  if mode=='cub':a=np.c_[a,x**3]
  return a
 def res(p, ids):
  a=design(u,p); return (a[ids]@ls(a[ids],v[ids])-v[ids]).ravel()
 errs=[]
 for ids in [np.arange(24),*np.array_split(np.random.default_rng(1).permutation(24),4)]:
  ii=ids if len(ids)==24 else np.setdiff1d(np.arange(24),ids)
  p=least_squares(res,{'locked':[2.85,-.4],'colfreq':[2.85]*4}.get(mode,[2.85]),args=(ii,)).x
  a=design(u,p); errs.append(np.mean(abs(a[ids]@ls(a[ii],v[ii])-v[ids])))
  if len(ids)==24: print(mode,'p',p, end=' ')
 print(np.round(errs,6))
base=aff(w); old=m['m']['_scalar_features'](w)
banks={}
for name,q in [('w',w),('z',z)]:
 cols={}
 for d in range(2,5):
  for i in combinations_with_replacement(range(4),d):cols[name+''.join(map(str,i))]=np.prod(q[:,i],axis=1)
 banks[name+' poly']=cols.copy()
 dirs={str(i):q[:,i] for i in range(4)}
 dirs.update({f'{i}+{c}*{j}':q[:,i]+c*q[:,j] for i,j in combinations(range(4),2) for c in (-1,-.5,.5,1)})
 for k,r in dirs.items():
  for c in (.5,1.,1.5,2.,3.):
   for namef,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda a:a/(1+a*a)),('inv',lambda a:1/(1+a*a))]:
    a=f(c*r); cols[f'{name}{namef}({c},{k})']=a
    if len(k)==1:
     for j in range(4): cols[f'{name}{j}*{namef}({c},{k})']=q[:,j]*a
 banks[name+' extended']=cols

def splits():
 yield np.arange(52),np.arange(52,72)
 for ix in np.array_split(np.random.default_rng(17).permutation(72),4): yield np.setdiff1d(np.arange(72),ix),ix
for bname,b in [('affine',base),('old',old)]:
 for name,cols in banks.items():
  names=list(cols); a=np.column_stack(list(cols.values())); errs=[]; sels=[]
  for it,iv in splits():
   q=np.linalg.qr(b[it])[0]; ar=a[it]-q@(q.T@a[it]); factor=np.linalg.norm(ar,axis=0); ar/=np.where(factor>1e-8,factor,np.inf); r=y[it]-q@(q.T@y[it]); sel=[]; e=[]
   for j in range(7):
    score=abs(ar.T@r); score[sel]=-1; sel.append(np.argmax(score)); d=np.c_[b,a[:,sel]]; beta=ls(d[it],y[it]); r=y[it]-d[it]@beta; e.append(np.mean(abs(d[iv]@beta-y[iv])))
   sels.append([names[k] for k in sel]); errs.append(e)
  print(bname,name,'gate',np.round(errs[0],4),'worst4',np.round(np.max(errs[1:],axis=0),4),'chosen',sels[0], 'replicated', [i[0] for i in sels[1:]])
for name,b in [('affine',base),('old',old),('quadratic',np.c_[base,*[w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)]])]:
 for space in (w,z,np.c_[z,w]):
  d=cdist(space,space,'sqeuclidean'); vals=[]
  for length in (.25,.5,1.,2.,4.):
   k=np.exp(-d/(2*length**2));
   for reg in (.0001,.001,.01,.1):
    e=[]
    for it,iv in splits():
     mat=k[np.ix_(it,it)]+reg*np.eye(len(it)); a=solve(mat,np.c_[b[it],y[it]],assume_a='pos'); co=ls(b[it].T@a[:,:-1],b[it].T@a[:,-1]); pr=b[iv]@co + k[np.ix_(iv,it)]@(a[:,-1]-a[:,:-1]@co); e.append(np.mean(abs(pr-y[iv])))
    vals.append((max(e[1:]),e[0],length,reg))
  print('kernel',name,space.shape[1],space is w, sorted(vals)[:3])
