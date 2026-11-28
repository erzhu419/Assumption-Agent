import itertools as it,pickle,numpy as np
from scipy.optimize import minimize_scalar
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['panel_b']; u,v=b['u'],b['v']; n,d=u.shape
A=np.c_[np.ones(n),u]; q=np.linalg.qr(A)[0]; P=np.eye(n)-q@q.T; R=P@v; Q=np.linalg.qr(R)[0];
D={}; args={}
for j in range(d):
 args[f'x{j}']=u[:,j]
 for k in range(j):
  for c in [-2,-1,-.5,.5,1,2]: args[f'x{j}{c:+g}x{k}']=u[:,j]+c*u[:,k]
  args[f'x{j}*x{k}']=u[:,j]*u[:,k]
for order in (2,3,4,5):
 for t in it.combinations_with_replacement(range(d),order): D[str(t)]=u[:,t].prod(1)
funcs={'sin':lambda x,s:np.sin(s*x),'cos':lambda x,s:np.cos(s*x),'tanh':lambda x,s:np.tanh(s*x),'rat':lambda x,s:x/(1+s*x*x),'rat2':lambda x,s:x*x/(1+s*x*x),'exp':lambda x,s:np.exp(-s*x*x),'soft':lambda x,s:np.logaddexp(s*x,-s*x),'atan':lambda x,s:np.arctan(s*x),'signed-exp':lambda x,s:x*np.exp(-s*x*x)}
for name,x in args.items():
 for fn,f in funcs.items():
  for s in np.geomspace(.18,8,28): D[(fn,name,s)]=f(x,s)
 D[('abs',name,1)]=abs(x)
def rank(target):
 F=P@np.column_stack(list(D.values())); H=F/np.maximum(np.linalg.norm(F,axis=0),1e-12); score=np.linalg.norm(target.T@H,axis=0)**2
 names=list(D); seen=set(); best=[]
 for i in np.argsort(score)[::-1]:
  k=names[i]; group=k[:2] if isinstance(k,tuple) else k
  if group in seen: continue
  seen.add(group)
  if isinstance(k,tuple) and k[0] in funcs:
   f=funcs[k[0]]; x=args[k[1]]
   def obj(s):
    y=P@f(x,s); return -np.sum((target.T@y)**2)/max(y@y,1e-24)
   opt=minimize_scalar(obj,bounds=(k[2]/1.3,k[2]*1.3),method='bounded'); k=(*group,round(opt.x,6)); sc=-opt.fun
  else: sc=score[i]
  best.append((k,round(sc,8)))
  if len(best)==12: break
 print(best)
rank(Q)
# Residual-space simultaneous pursuit: tells whether a short shared linear dictionary suffices.
keys=list(D); F=P@np.column_stack(list(D.values())); H=F/np.maximum(np.linalg.norm(F,axis=0),1e-12); Z=A.copy(); E=R.copy()
for step in range(10):
 q=np.linalg.qr(Z)[0]; H2=H-q@(q.T@H); scores=np.linalg.norm(H2.T@E,axis=1)/np.maximum(np.linalg.norm(H2,axis=0),1e-9); i=np.argmax(scores); Z=np.c_[Z,D[keys[i]]]; E=v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]; print('pursuit',keys[i],np.sqrt(np.mean(E**2)))
