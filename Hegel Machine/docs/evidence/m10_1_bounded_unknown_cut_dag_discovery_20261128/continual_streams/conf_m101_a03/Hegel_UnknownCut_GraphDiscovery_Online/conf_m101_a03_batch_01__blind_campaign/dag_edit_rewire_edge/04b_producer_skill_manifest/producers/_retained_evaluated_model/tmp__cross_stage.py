exec(open('/tmp/readout_probe.py').read().split("print('ranges'")[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
A=affine(z); inds=np.arange(n); ix=inds[:52]; v=inds[52:]; fs=[v]+list(folds[0])+[np.argsort(q[:,i])[sl] for q in (x,z) for i in range(4) for sl in (slice(0,12),slice(-12,None))]
pairs=list(combinations_with_replacement(range(4),2)); U=np.column_stack([z[:,i]*z[:,j] for i,j in pairs]); B=U[:,[2,6]]
D={f'{s}{i}':q[:,i] for s,q in [('x',x),('z',z)] for i in range(4)}
def fit(m,Q,op=np.tanh):
 p0=np.linalg.lstsq(A[m],y[m],rcond=None)[0]
 def design(p,I):
  u=A[I]@p; return np.column_stack([np.ones(len(I)),u,op(u),Q[I]])
 def cost(p):
  F=design(p,m);return y[m]-F@np.linalg.lstsq(F,y[m],rcond=None)[0]
 r=min((least_squares(cost,p0*s,max_nfev=100) for s in [.65,1.3]),key=lambda r:r.cost)
 F=design(r.x,inds);c=np.linalg.lstsq(F[m],y[m],rcond=None)[0];return F@c,r.x,c,F
pred,p,c,F=fit(ix,B)
J=np.column_stack([F,A*(c[1]+c[2]/np.cosh(A@p)**2)[:,None]])
proj=A@p
atoms={}
for i in range(4):
 for j in range(4):
  xx=x[:,i];zz=z[:,j];atoms[f'x{i}*z{j}']=xx*zz; atoms[f'x{i}*x{j}']=xx*x[:,j]
  for s in [.5,1.,1.5,2.]:
   for f in [np.sin,np.tanh,np.cos]:
    atoms[f'x{i}*{f.__name__}({s}*z{j})']=xx*f(s*zz)
    atoms[f'x{i}*{f.__name__}({s}*x{j})']=xx*f(s*x[:,j])
    for k in range(j+1,4):
     for sig in [-1,1]:
      atoms[f'x{i}*{f.__name__}({s}*(z{j}+{sig}z{k}))']=xx*f(s*(zz+sig*z[:,k]))
 for s in [.5,1.,1.5,2.]:
  for f in [np.sin,np.tanh,np.cos]: atoms[f'x{i}*{f.__name__}({s}*score)']=x[:,i]*f(s*proj)
names=list(atoms); E=np.column_stack(list(atoms.values())); R=E[ix]-J[ix]@np.linalg.lstsq(J[ix],E[ix],rcond=1e-9)[0]; score=np.abs((y-pred)[ix]@R)/np.maximum(np.linalg.norm(R,axis=0),1e-10)
print('best atoms:',[(names[j],round(score[j],5)) for j in np.argsort(score)[-16:][::-1]])
for nm,Q in [('full',U),('pair',B)]+[(names[j],np.column_stack([B,E[:,j]])) for j in np.argsort(score)[-16:][::-1]]+[(f'full plus {names[j]}',np.column_stack([U,E[:,j]])) for j in np.argsort(score)[-5:][::-1]]:
 out,p,c,_=fit(ix,Q); errs=[]
 for val in fs:
  po,*_=fit(np.setdiff1d(inds,val),Q);errs.append(np.mean(abs(po[val]-y[val])))
 print(nm,np.round([np.mean(abs(out[ix]-y[ix])),errs[0],max(errs[1:7]),max(errs[7:])],5),np.round(c,4))
