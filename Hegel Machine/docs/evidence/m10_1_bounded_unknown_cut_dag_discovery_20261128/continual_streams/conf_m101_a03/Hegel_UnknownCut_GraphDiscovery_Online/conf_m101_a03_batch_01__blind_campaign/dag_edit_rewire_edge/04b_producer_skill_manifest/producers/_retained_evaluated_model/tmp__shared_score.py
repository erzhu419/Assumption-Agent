exec(open('/tmp/readout_probe.py').read().split("print('ranges'")[0])
from scipy.optimize import least_squares
A=affine(z); U=poly(z,2)[:,5:]; pair=[(i,j) for i in range(4) for j in range(i,4)]; V=np.column_stack([x[:,i]*z[:,j] for i in range(4) for j in range(4)])
D={'full':U,'cross':U[:,[pair.index((0,2)),pair.index((1,3))]],'xz':V}
trans={'tanh':np.tanh,'rat':lambda u:u/(1+u*u),'soft':lambda u:u/np.sqrt(1+u*u),'atan':np.arctan,'asinh':np.arcsinh}
def fit(m,B,op,shared=False):
 p0=np.linalg.lstsq(A[m],y[m],rcond=None)[0]; f=trans[op]
 def design(p,I):
  aa=A[I]; u=aa@p; return np.column_stack([np.ones(len(u)),u,f(u),B[I]]) if shared else np.column_stack([aa,B[I],f(u)])
 def cost(p):
  G=design(p,m);return y[m]-G@np.linalg.lstsq(G,y[m],rcond=None)[0]
 p=min((least_squares(cost,np.clip(p0*s,-4,4),bounds=(-5,5),max_nfev=140) for s in [.5,1.,2.]),key=lambda r:r.cost).x
 G=design(p,np.arange(n)); c=np.linalg.lstsq(G[m],y[m],rcond=None)[0];return G@c,p,c
idx=np.arange(52); mt=np.arange(52,n)
fs=[mt]+list(folds[0])+[np.argsort(z[:,i])[:12] for i in range(4)]+[np.argsort(z[:,i])[-12:] for i in range(4)]
for bn in D:
 for op in trans:
  for shared in [True,False]:
   if bn=='xz' and not shared: continue
   vals=[]
   for v in fs:
    pred,p,c=fit(np.setdiff1d(np.arange(n),v),D[bn],op,shared); vals.append(np.mean(abs(pred[v]-y[v])))
   pred,p,c=fit(idx,D[bn],op,shared)
   print(bn,op,shared,np.round([np.mean(abs(pred[idx]-y[idx])),vals[0],np.mean(vals[1:7]),max(vals[1:7]),max(vals[7:])],5),np.round(p,3),np.round(c,3),flush=True)
print('monomials',pair)
