exec(open('/tmp/probe.py').read().split('for nm,x')[0])
from scipy.optimize import minimize_scalar
from scipy.special import expit
c=s['panel_c']; u=c['u']; c1=lambda x:one(powers(x,2))@ls(one(powers(u,2)),c['v1']);
fns={'cos':np.cos,'rat':lambda x:1/(1+x*x),'cosh':lambda x:1/np.cosh(x),'exp':lambda x:np.exp(-x*x),'log':lambda x:np.log1p(x*x),'sqrt':lambda x:np.sqrt(1+x*x),'swish':lambda x:x*expit(x),'softplus':lambda x:np.logaddexp(0,x)}
cf={}
for name,fn in fns.items():
 F=lambda x,k:one(np.c_[x,fn(k*x)]); obj=lambda k:np.std(F(u,k)@ls(F(u,k),c['v2'])-c['v2']); opt=minimize_scalar(obj,bounds=(.04,3),method='bounded'); cf[name]=(opt.fun,opt.x,fn);print(name,opt.fun,opt.x,obj(1.25))
_,k,fn=min(cf.values(),key=lambda q:q[0]);F=lambda x:one(np.c_[x,fn(k*x)]);c2=lambda x:F(x)@ls(F(u),c['v2']); bb=lambda x:one(x)@B
atoms={}; scored=[]
for nm,x0 in [('z',z),('w',w),('raw',s['train']['x'])]:
 for a,func in [('',lambda x:x),('P',c1),('Q',c2),('PQ',lambda x:c2(c1(x))),('QP',lambda x:c1(c2(x)))]:
  x=func(x0);atoms[nm+a]=x
  if a:atoms[nm+a+'B']=bb(x)
for key,A in atoms.items():
 for key2,C in atoms.items():
  pair=(A[:,:,None]*C[:,None,:]); dot=np.trace(pair,axis1=1,axis2=2);diag=np.diagonal(pair,axis1=1,axis2=2)
  for nm,f in [('lin',np.c_[A,C]),('dot',np.c_[A,C,dot]),('diag',np.c_[A,C,diag]),('full',np.c_[A,C,pair.reshape(len(A),-1)]),('q',np.c_[A,C,powers(A,2),powers(C,2)]),('qdot',np.c_[A,C,powers(A,2),powers(C,2),dot])]:
   f=one(f);m=ls(f[:52],np.c_[y,tr][:52]);r=f[52:]@m-np.c_[y,tr][52:]; rr=np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]));scored.append((rr,key,key2,nm))
for i in [0,1,2]:
 print('best',i);print(*sorted(scored,key=lambda t: t[0][i] if i<2 else max(t[0]))[:18],sep='\n')
print('P coefficients',ls(one(powers(u,2)),c['v1']));print('Q coefficients',ls(F(u),c['v2']));
