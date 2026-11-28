exec(open('/tmp/probe.py').read().split('res=[]')[0])
u=c['u']; v=c['v1']; print('u and v',np.c_[u,v].round(3),flush=True)
from itertools import combinations,product
from scipy.optimize import minimize_scalar
from scipy.special import erf
pairs=list(combinations(range(4),2)); seqs=[np.eye(4), B[1:]/.88,B[1:].T/.88]; bases={}
for j,Q in enumerate(seqs):
 h=u@Q
 for key,k in [('coord',h),('sum',np.stack([h[:,i]+h[:,j] for i,j in pairs],1)),('dif',np.stack([h[:,i]-h[:,j] for i,j in pairs],1)),('prod',np.stack([h[:,i]*h[:,j] for i,j in pairs],1)),('diffsq',np.stack([h[:,i]**2-h[:,j]**2 for i,j in pairs],1))]:
  for k1,col in enumerate(k.T): bases[f'{j}:{key}:{k1}']=col
H=np.stack(list(bases.values()),1); names=np.array(list(bases)); n=len(u)
funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'soft':lambda z:z/(1+abs(z)),'orat':lambda z:z/(1+z*z),'erat':lambda z:1/(1+z*z),'gauss':lambda z:np.exp(-z*z),'abs':np.abs,'sq':np.square,'identity':lambda z:z}
L=one(u); Q=np.linalg.qr(L)[0]; proj=np.eye(n)-Q@Q.T; C=[]; labels=[]
for name,fn in funcs.items():
 for a in np.arange(.25,6.01,.25) if name not in ('abs','sq','identity') else [1]:
  for p in [0.,-.75,.75] if name not in ('sq','identity') else [0]:
   C.append(fn(a*H+p)); labels.extend([(name,round(a,2),p,q) for q in names])
for a in [.15,.35,.6,1.25,2.5,4.]:
 for h in range(16):
  C.append(u[:,h//4,None]/(1+a*u[:,h%4,None]**2)*u); labels.extend([('p/q',a,h//4,(h%4,j)) for j in range(4)])
C=np.column_stack(C)
for pr in [np.eye(n)-1/n,proj]:
 F=pr@C; F/=np.maximum(np.linalg.norm(F,axis=0),1e-12); V=pr@v
 for j in range(4):
  rr=V[:,j]; ss=abs(F.T@rr); ids=np.argsort(-ss)[:8]; print('matches',j,[(round(ss[i]/np.linalg.norm(rr),4),labels[i]) for i in ids],flush=True)
  Id=[]; rms=[]
  for k in range(4):
   idx=np.argmax(abs(F.T@rr)); Id.append(idx); sub=F[:,Id]; rr=V[:,j]-sub@lin(sub,V[:,j]); rms.append(np.sqrt((rr*rr).mean()))
  print('greedy',[(labels[i],round(m,5)) for i,m in zip(Id,rms)])
from itertools import combinations_with_replacement
for deg in [1,2,3]:
 P=np.c_[np.ones(n),*[np.prod(u[:,ij],axis=1) for d0 in range(1,deg+1) for ij in combinations_with_replacement(range(4),d0)]]
 if P.shape[1]>=n: break
 K=P@np.linalg.pinv(P); R=(K@v-v)/(1-K.diagonal()[:,None]); print('poly loo',deg,np.sqrt(np.mean(R**2,axis=0)))
