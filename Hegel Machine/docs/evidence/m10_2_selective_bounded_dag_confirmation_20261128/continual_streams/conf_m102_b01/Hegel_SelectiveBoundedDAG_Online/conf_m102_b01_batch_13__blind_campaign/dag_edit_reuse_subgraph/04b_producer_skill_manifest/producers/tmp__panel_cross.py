from panel_probe import *
from itertools import combinations, combinations_with_replacement

def assess(name,fn,verbose=False):
 A=aff(fn(u)); D=aff(fn(x)); q=ls(A,v); r=A@q-v
 loo=r/np.maximum(1-np.sum(A*np.linalg.pinv(A).T,axis=1),1e-8)[:,None]
 print(name, A.shape[1], 'rms/loo',np.sqrt(np.mean(r*r)),np.sqrt(np.mean(loo*loo)),'head CV',score(D@q),flush=True)
 if verbose: print(np.round(q,4));print('residual per output',np.sqrt(np.mean(r*r,axis=0)))

def prod(z,pairs):return np.stack([np.prod(z[:,ix],axis=1) for ix in pairs],axis=1)
a2=list(combinations_with_replacement(range(4),2)); p2=list(combinations(range(4),2)); p3=list(combinations(range(4),3)); a3=list(combinations_with_replacement(range(4),3)); arch=[(0,1),(0,3),(1,2),(2,3)]
assess('linear',lambda z:z)
assess('all quadratic',lambda z:np.c_[z,prod(z,a2)],True)
assess('cross terms only',lambda z:prod(z,p2),True)
assess('quadratics only',lambda z:prod(z,a2))
assess('archive 4 cross',lambda z:prod(z,arch),True)
assess('pair+triple',lambda z:np.c_[prod(z,p2),prod(z,p3)])
assess('linear + odd powers + pair',lambda z:np.c_[z,z**3,prod(z,p2)])
assess('quadratic + triples',lambda z:np.c_[z,prod(z,a2),prod(z,p3)])
for f in [np.sin,np.tanh,np.arctan,np.arcsinh]:
 for g in [1,2.15,3,5]:
  assess(f.__name__+str(g)+' cross of f(x)',lambda z:prod(f(g*z),p2))
  assess(f.__name__+str(g)+' f(cross)',lambda z:f(g*prod(z,p2)))
