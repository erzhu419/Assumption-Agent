exec(open('/tmp/sparse_probe.py').read().split("print('coefficients")[0])
from itertools import combinations_with_replacement as cr
splits=[(np.arange(52),np.arange(52,72))];
for seed in range(6):splits+=list(KFold(6,shuffle=True,random_state=seed).split(F))
r=np.linalg.norm(w,axis=1); splits += [(np.argsort(r)[:-12],np.argsort(r)[-12:])]
def stats(e): return np.round([e[0],np.mean(e[1:-1]),max(e[1:-1]),e[-1]],6)
def err(builder):return stats([np.abs(builder(a,b)-y[b]).mean() for a,b in splits])
Q=F[:,5:]; products=Q[:,[pairs.index(k) for k in [(0,1),(0,3),(1,2),(2,3)]]]; c=s['panel_c']; uc=c['u']; C=np.column_stack([np.ones(len(uc)),*[uc[:,i]*uc[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]]]); V=np.c_[np.ones(len(w)),products]@ls(C,c['v1']);
for name,A in [('quadratic',F),('even',F[:,[0,*range(5,15)]]),('pair',products),('quartic',np.c_[F[:,[0,*range(5,15)]],*[np.prod(w[:,ix],axis=1) for ix in cr(range(4),4)]]),('pair-cubic',np.c_[np.ones(len(w)),products,*[np.prod(products[:,ix],axis=1) for ix in cr(range(4),3)]]),('pair-quadratic',np.c_[np.ones(len(w)),products,*[np.prod(products[:,ix],axis=1) for ix in cr(range(4),2)]])]:
 for l in ([0,.02,.1,.5] if A.shape[1]>15 else [0]):
  def pred(a,b):
   zz=A[a]; sd=np.sqrt(np.mean(zz**2,axis=0)); sd=np.maximum(sd,1e-8); zz=zz/sd; return (A[b]/sd) @ ls(np.r_[zz,np.sqrt(l)*np.eye(A.shape[1])],np.r_[y[a],np.zeros(A.shape[1])])
  print(name,l,err(pred))
for kind,fn in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('rational',lambda x:x/(1+x*x)),('saturate',lambda x:x/np.sqrt(1+x*x))]:
 for g in [.1,.25,.5,1,2]:
  a0=np.c_[F,fn(g*V)];print(kind,g,err(lambda a,b:a0[b]@ls(a0[a],y[a])))
print('y residual sorted by y: ',[(np.round(y[i],3),np.round((y-F@ls(F,y))[i],3)) for i in np.argsort(y)[::3]])
