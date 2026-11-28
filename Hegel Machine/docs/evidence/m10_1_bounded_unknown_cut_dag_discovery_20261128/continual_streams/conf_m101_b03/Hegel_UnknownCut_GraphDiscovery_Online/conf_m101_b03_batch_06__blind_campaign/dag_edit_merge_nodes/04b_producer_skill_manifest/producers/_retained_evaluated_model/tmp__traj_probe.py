from probe import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
X=S['train']['x']; y=S['train']['y']; T=S['train']['traj']; t=S['eval_times']; z,c,d,r=panels(S,X)
u=S['panel_b']['u']; b=aff(z)@ls(aff(u),S['panel_b']['v'])
_,sing,V=np.linalg.svd(T,full_matrices=False)
print('temporal vectors',np.round(V[:2],5)); print('rank-2 MAE',np.mean(abs(T-T@V[:2].T@V[:2])))
for second in ['t','exp2','exp_ratio2','t_exp','sin','one']:
 def cols(k):
  g=-np.expm1(-k*t)
  h={'t':t,'exp2':-np.expm1(-2*k*t),'exp_ratio2':-np.expm1(-.5*k*t),'t_exp':t*np.exp(-k*t),'sin':np.sin(t),'one':np.ones_like(t)}[second]
  return np.c_[g,h]
 def cost(k):
  A=cols(k); return np.mean((A@ls(A,T.T)-T.T)**2)**.5
 opt=minimize_scalar(cost,bounds=(.02,4),method='bounded'); print('time',second,round(opt.x,6),round(opt.fun,7))
base=np.c_[z,c,d,c*d]
for q,w in [('z',z),('b',b),('c',c),('d',d),('cd',c*d),('x',X)]:
 for p,v in [('z',z),('b',b),('c',c),('d',d),('cd',c*d),('x',X)]:
  if q>p: continue
  for typ,f in [('same',w*v),('outer',(w[:,:,None]*v[:,None,:]).reshape(len(z),-1))]:
   A=np.c_[base,f]; err=cv(A,T); e=np.round(err.mean(1),4); tr=np.mean(abs(aff(A)@ls(aff(A),T)-T));
   print(q,p,typ,'rank',np.linalg.matrix_rank(aff(A),tol=1.e-8),'cv',e,'fit',round(tr,5))
# Pursuit on residualized candidate interactions, with selection repeated independently inside each fold.
F={}
VARS={'z':z,'b':b,'c':c,'d':d,'x':X,'cd':c*d}
for a,A in VARS.items():
 for bname,B in VARS.items():
  if a>bname:continue
  for i in range(4):
   for j in range(4):
    if a!=bname or j>=i:F[f'{a}{i}*{bname}{j}']=A[:,i]*B[:,j]
for n,A in VARS.items():
 for kind,W in [('tanh',np.tanh(A)),('sat',A/(1+A*A)),('cube',A**3),('sin',np.sin(A)),('cos',np.cos(A))]:
  for j in range(4):F[f'{kind}({n}{j})']=W[:,j]
for A,n in [(base,'base'),(np.c_[base,z*c],'base+zc')]:
 fm=np.column_stack(list(F.values())); keys=list(F)
 for i in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,52),np.array([],dtype=int)]:
  j=np.setdiff1d(np.arange(len(T)),i); W=aff(A); R=fm[j]-W[j]@ls(W[j],fm[j]); scales=np.linalg.norm(R,axis=0); valid=scales>1e-7; active=[]; scores=[]
  for step in range(10):
   res=T[j]-W[j]@ls(W[j],T[j]); cur=fm[j]-W[j]@ls(W[j],fm[j]); norms=np.linalg.norm(cur,axis=0); good=valid & (norms>1e-7); score=np.where(good,np.linalg.norm(cur.T@res,axis=1)/np.maximum(norms,1e-7),-1); k=int(np.argmax(score)); active.append(keys[k]); valid[k]=False; W=np.c_[W,fm[:,k]]; pred=W@ls(W[j],T[j]); scores.append(round(np.mean(abs(pred[i if len(i) else j]-T[i if len(i) else j])),5))
  print('pursuit',n,'fold',i[0] if len(i) else 'fit',scores,active)
