from shape04 import *
from itertools import combinations, combinations_with_replacement
order=np.random.default_rng(194).permutation(72); folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]
# Find omitted terms in reconstructed panel-B input coordinates.
base=list(candidates())[0][1]; p0=[.55,2.4,-.095]; lo0=[.015,.6,-4]; hi0=[8,8,4]
p=least_squares(lambda p:(aff(base(u,p))@ls(aff(base(u,p)),v)-v).ravel(),p0,bounds=(lo0,hi0)).x
A=aff(base(u,p)); B=ls(A,v); R=(v-A@B)@np.linalg.inv(B[1:]); print('unmixed panel residual correlations')
for i in range(4):
 ds={}
 for j,k in combinations_with_replacement(range(4),2):
  for n,f in [('uxu',u[:,j]*u[:,k]),('fxf',base(u,p)[:,j]*base(u,p)[:,k]),('uxf',u[:,j]*base(u,p)[:,k]),('fxu',base(u,p)[:,j]*u[:,k])]: ds[f'{n}{j}{k}']=f
 for j in range(4):
  for n,f in [('sin',np.sin(2*u[:,j])),('cos',np.cos(2*u[:,j])),('lin',u[:,j]),('abs',abs(u[:,j]))]: ds[f'{n}{j}']=f
 out=[]
 for n,f in ds.items():
  g=f-A@ls(A,f); out.append((abs(np.corrcoef(g,R[:,i])[0,1]),n))
 print(i,sorted(out,reverse=True)[:10])
for k in (1,2,3):
 def roll(q):return np.roll(q,-k,axis=1)
 for n,g in [('cross',lambda q,f:q*roll(q)),('smooth cross',lambda q,f:q*roll(f)),('cross cube',lambda q,f:q*q*roll(q)),('cross cube reverse',lambda q,f:q*roll(q)**2),('mix before',None),('even',lambda q,f:q*q),('sin mix',lambda q,f:np.sin(2*roll(q))),('linear neighbor',lambda q,f:roll(q))]:
  f=(lambda q,p:base(q+p[3]*roll(q),p)) if g is None else (lambda q,p:base(q,p)+p[3]*g(q,base(q,p)))
  out=[]
  for i,j in folds:
   pred,p,r=solve_shape(f,p0+[0],(lo0+[-.6],hi0+[.6]),i,2); out.append(scores(i,j)(pred(w[j])))
  print(k,n,np.round(p,4),'panel',round(np.sqrt(np.mean(r[:v.size]**2)),6),'56/16',np.round(out[0],5),'max',np.round(np.max(out,axis=0),5))
