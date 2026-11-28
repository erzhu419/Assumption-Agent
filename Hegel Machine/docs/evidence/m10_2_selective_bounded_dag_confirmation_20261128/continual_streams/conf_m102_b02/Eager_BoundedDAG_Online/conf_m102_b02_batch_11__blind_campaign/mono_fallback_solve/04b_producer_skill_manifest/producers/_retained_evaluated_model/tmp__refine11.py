import numpy as np
from itertools import combinations,product
from scipy.linalg import qr
q=np.load('/tmp/d11.npz'); b=q['b']; Y=np.c_[q['y'],q['T']]; n=len(Y); rng=np.random.default_rng(811); splits=[np.arange(n)]+[rng.permutation(n) for _ in range(20)]+[np.argsort(np.linalg.norm(b,axis=1))]; ij=[(p[:52],p[52:]) for p in splits]; pairs=list(combinations(range(4),2)); tele=2.15

def err(F,reg=0):
 G=F@F.T; E=[]
 for i,j in ij:
  if reg: p=G[np.ix_(j,i)]@np.linalg.solve(G[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i]);
  else: p=F[j]@np.linalg.lstsq(F[i],Y[i],rcond=1e-8)[0]
  e=abs(p-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(E)

def mats(seq):return np.column_stack(seq)

def gram(k,variant):
 h=np.arcsinh(k*b); g=np.c_[.4*b,h]; R=np.c_[tuple(b[:,i]*b[:,j] for i,j in pairs)]
 A=[np.ones(n),b,h]; diag=[b*b,b*h,h*h]; cross=[mats([u[:,i]*v[:,j] for i,j in pairs]+([v[:,i]*u[:,j] for i,j in pairs] if l==1 else [])) for l,(u,v) in enumerate([(b,b),(b,h),(h,h)])]; cc=[mats([np.arcsinh(k*(b[:,i]+sg*b[:,j])) for i,j in pairs for sg in (1,-1)]),mats([b[:,i]*b[:,j]/(1+.22*(b[:,i]-sg*b[:,j])**2) for i,j in pairs for sg in (1,-1)]),mats([h[:,i]*h[:,j]/(1+.22*(b[:,i]-sg*b[:,j])**2) for i,j in pairs for sg in (1,-1)])];
 if variant=='state': return np.c_[tuple(A+diag+cross)]
 if variant=='mono':return np.c_[tuple(A+diag)]
 if variant=='sparse':return np.c_[tuple(A+diag+[cross[-1]])]
 if variant=='coupling':return np.c_[tuple(A+diag+cc)]
 if variant=='sum':return np.c_[tuple(A+diag+cc[:1])]
 if variant=='rat':return np.c_[tuple(A+diag+cc[1:2])]
 if variant=='rath':return np.c_[tuple(A+diag+cc[2:])]
 if variant=='archive':return np.c_[np.ones(n),g/2,mats([g[:,i]*g[:,j]/(8 if i==j else np.sqrt(32)) for i in range(8) for j in range(i,8)])]
 if variant=='honly':return np.c_[np.ones(n),h,h*h,cross[-1]]
 if variant=='proj':return np.c_[tuple(A+diag+[cross[1]])]
 if variant=='triple':return np.c_[tuple(A+diag+[cross[-1],mats([h[:,i]*h[:,j]*h[:,l] for i,j,l in combinations(range(4),3)])])]
 if variant=='saturate':return np.c_[tuple(A+diag+[b/(1+.22*b*b),1/(1+.22*b*b),cross[-1]])]

a=[]
for k,v,reg in product([.5,1,1.5,2,2.15,2.5,3,4,6],['state','mono','sparse','coupling','sum','rat','rath','archive','honly','proj','triple','saturate'],[0,1e-6,.0001,.003]):
 F=gram(k,v); E=err(F,reg); a.append((E,k,v,reg))
for l in (0,1):
 for crit in (lambda e:e[1:-1,l].mean(),lambda e:e[-1,l]):
  print('HEAD',l)
  for E,k,v,reg in sorted(a,key=lambda r:crit(r[0]))[:10]:print(k,v,reg,np.round([E[0,l],E[1:-1,l].mean(),E[1:-1,l].max(),E[-1,l]],5))
for k,v in [(2.15,'archive'),(2.15,'sparse'),(2.15,'state')]:
 F=gram(k,v); c=np.linalg.lstsq(F,Y,rcond=1e-8)[0]; print(k,v,'fit',np.abs(F@c-Y).mean(0),'first9',c[:9, [0,8]].round(3))
