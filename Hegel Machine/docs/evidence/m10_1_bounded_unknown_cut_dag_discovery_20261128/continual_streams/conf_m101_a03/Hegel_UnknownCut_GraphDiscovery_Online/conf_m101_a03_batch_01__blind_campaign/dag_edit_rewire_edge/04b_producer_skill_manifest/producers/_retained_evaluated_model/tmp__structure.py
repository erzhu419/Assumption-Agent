exec(open('/tmp/readout_probe.py').read().split("print('ranges'")[0])
from scipy.linalg import qr
from scipy.optimize import minimize_scalar

def rm(f,g): return g-f@np.linalg.lstsq(f,g,rcond=None)[0]
base=affine(z); yr=rm(base,y)
print('operator',np.round(np.linalg.lstsq(affine(x),z,rcond=None)[0],4));print('noise candidates',np.sort(np.abs(yr))[-10:]);print('trajectory sv',np.linalg.svd(t,compute_uv=False))
dirs={f'{n}{i}':q[:,i] for n,q in [('x',x),('z',z)] for i in range(4)}
pairs={f'{n}{sg}{m}':dirs[n]+sgn*dirs[m] for k,n in enumerate(dirs) for m in list(dirs)[k+1:] for sg,sgn in [('+',1),('-',-1)]}
amps={'1':np.ones(len(x)),**dirs}; atoms=[]
for nm,d in (dirs|pairs).items():
 for f in [.5,1,1.5,2,2.5,3,4]:
  for an,a in amps.items():
   for fn,h in [('sin',np.sin(f*d)),('cos',np.cos(f*d)),('tanh',np.tanh(f*d)),('rat',d/(1+(f*d)**2)),('sq',d*d),('cube',d*d*d)]:
    if fn in ['sq','cube'] and f!=1: continue
    atoms.append((f'{an}*{fn}({f}*{nm})',a*h))
for n,d in dirs.items():
 for m,b in dirs.items():
  for k,c in dirs.items():
   atoms.append((f'{n}*{m}*{k}',d*b*c))
n=len(y)
D=np.column_stack([a for _,a in atoms]); F=base.copy();
for i in range(9):
 R=rm(F,D); r=rm(F,y); norms=np.linalg.norm(R,axis=0); s=np.abs(r@R)/np.maximum(norms,1e-10); order=np.argsort(s)[-6:][::-1]; j=order[0]; F=np.column_stack([F,D[:,j]]);print('step',i,[(atoms[k][0],round(s[k],4)) for k in order],cv(F)[[0,2,4]])
for nm,q in [('x',x),('z',z)]:
 F=poly(q,3)
 for lam in [.0001,.001,.01,.05,.1,.2,.5,1]:
  fs=np.concatenate(folds); err=[]
  for v in fs:
   m=np.ones(n,bool);m[v]=False; err.append(np.mean(abs(y[v]-F[v]@np.linalg.solve(F[m].T@F[m]+lam*np.diag([0]*5+[1]*30),F[m].T@y[m]))))
  print('ridge',nm,lam,round(np.mean(err),5),round(max(err),5))
