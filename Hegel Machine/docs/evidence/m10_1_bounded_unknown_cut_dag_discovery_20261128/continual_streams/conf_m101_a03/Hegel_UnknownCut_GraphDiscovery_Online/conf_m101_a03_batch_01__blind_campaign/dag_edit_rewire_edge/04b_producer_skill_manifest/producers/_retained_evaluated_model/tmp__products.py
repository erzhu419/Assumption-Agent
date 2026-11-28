exec(open('/tmp/readout_probe.py').read().split("print('ranges'")[0])
from itertools import combinations
base=affine(z)
def rm(f,g): return g-f@np.linalg.lstsq(f,g,rcond=None)[0]
q={f'{s}{i}':a[:,i] for s,a in [('x',x),('z',z)] for i in range(4)}
prod=[(f'{a}*{b}',q[a]*q[b]) for a,b in combinations(q,2)]
D=[]; names=[]
def add(name,v): names.append(name); D.append(v)
for name,p in prod:
 for f in [.5,1.,1.5,2.,2.5,3.]:
  for tag,H in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda u:u/(1+u*u))]: add(f'{tag}({f}*{name})',H(f*p))
 for k,v in q.items():
  for f in [.5,1.,1.5,2.,2.5,3.]:
   for tag,H in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda u:u/(1+u*u))]: add(f'{name}*{tag}({f}*{k})',p*H(f*v))
  add(f'{name}*{k}',p*v)
  add(f'{name}*{k}^2',p*v*v)
  for f in [.5,1.,2.]:
   add(f'{name}/(1+{f}*{k}^2)',p/(1+f*v*v))
 for lab,v in prod:
  for tag,H in [('sin',np.sin),('tanh',np.tanh)]: add(f'{name}*{tag}({lab})',p*H(v))
  add(f'{name}/(1+({lab})^2)',p/(1+v*v))
D=np.column_stack(D); F=base.copy()
for i in range(4):
 R=rm(F,D); r=rm(F,y); norms=np.linalg.norm(R,axis=0); score=np.abs(r@R)/np.maximum(norms,1e-10); order=np.argsort(score)[-12:][::-1]; j=order[0]; F=np.column_stack([F,D[:,j]]);print('step',i,[(names[k],round(score[k],4)) for k in order],cv(F)[[0,2,4]])
