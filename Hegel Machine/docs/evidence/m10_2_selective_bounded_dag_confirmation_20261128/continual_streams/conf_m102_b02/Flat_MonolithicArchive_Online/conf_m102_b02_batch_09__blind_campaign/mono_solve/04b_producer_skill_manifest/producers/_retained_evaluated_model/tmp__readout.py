from panel import *
from scipy.optimize import root,minimize_scalar
from itertools import combinations,combinations_with_replacement
u,v=s['panel_a']['u'],s['panel_a']['v']; m=np.abs(v).max(1)<3; u,v=u[m],v[m]
def F(x):return .18*np.tanh(1.8*x)-.32*np.sin(2.4*np.roll(x,2,axis=-1))
a=np.array([ls(I(u[:,j]),(v-F(u))[:,j]) for j in range(4)])
x=np.r_[s['train']['x'],s['test']['x']]
z=np.array([root(lambda w:a[:,0]+a[:,1]*w+F(w)-r,(r-a[:,0])/a[:,1]).x for r in x]);print('A rmse',np.std(a[:,0]+a[:,1]*u+F(u)-v))
p=s['panel_b'];bc=ls(I(p['u']),p['v']);b=lambda x:I(x)@bc
p=s['panel_c'];u=p['u'];v=p['v1']; funcs.update(log=lambda x:np.logaddexp(x,-x), ex=lambda x:np.expm1(-x*x), cube=lambda x:x**3, ars=lambda x:np.arcsinh(2.15*x))
def make(q):
 aq=np.arcsinh(2.65*q);return I(np.c_[q,aq,q/(1+(5.47*q)**2)])
c=lambda x:make(x)@ls(make(u),v)
_,_,d=op(u,p['v2'],'sqrt','both')
coords={'z':z,'x':x,'b':b(z),'c':c(z),'d':d(z),'bc':c(b(z)),'bd':d(b(z)),'cb':b(c(z)),'db':b(d(z))}
t=s['eval_times'];tr=s['train']['traj'];y=s['train']['y'];trt=np.c_[y,tr]
splits=[(np.arange(52),np.arange(52,72))];rng=np.random.default_rng(802)
for _ in range(4):
 ix=rng.permutation(72);splits.append((ix[:52],ix[52:]))
def eval(D,alpha=.003,reg=None):
 if D.ndim==1:D=D[:,None]
 E=[]
 for ii,jj in splits:
  X=D[ii]; mu=X.mean(0); sd=np.maximum(X.std(0),1e-8); U=(X-mu)/sd; V=(D[jj]-mu)/sd
  yy=trt[ii]; ym=yy.mean(0); C=np.linalg.solve(U.T@U+alpha*np.eye(U.shape[1]),U.T@(yy-ym)); er=np.abs(V@C+ym-trt[jj]); E.append((er[:,0].mean(),er[:,1:].mean(),er[:,-1].mean()))
 return np.array(E)
def display(records):
 for k in (0,1):
  print('ENDPOINT',k)
  for name,e in sorted(records,key=lambda a:a[1][:,k].mean())[:18]:print(name,e[0].round(4),e.mean(0).round(4),e.max(0).round(4))
def P2(A):return np.column_stack([A[:,i]*A[:,j] for i,j in combinations_with_replacement(range(A.shape[1]),2)])
def feat(A):return np.c_[A,np.tanh(2.15*A),np.arcsinh(2.15*A),A**2,A**3,np.sin(2.15*A),A/(1+.22*A*A)]
if __name__=='__main__':
 print('stats',y.std(),tr.std(0),np.linalg.svd(tr,compute_uv=False)); rec=[]
 for k,A in coords.items():
  for nm,ft in [('lin',A),('p2',np.c_[A,P2(A)]),('nlin',feat(A))]:rec.append((k+nm,eval(ft)))
 for (k,A),(l,B) in combinations(coords.items(),2):
  for nm,ft in [('lin',np.c_[A,B]),('diag',np.c_[A,B,A*B]),('sq',np.c_[A,B,A*A,B*B,A*B]),('outer',np.c_[A,B,(A[:,:,None]*B[:,None,:]).reshape(len(A),-1)]),('sum',np.c_[A,B,np.sum(A*B,1)])]:rec.append((k+l+nm,eval(ft)))
 for keys in [('z','c','d'),('z','bc','bd'),('z','b','bc','bd'),('z','c','bd'),('z','bc','d'),('z','c','d','bc','bd')]:
  ft=np.column_stack([coords[k] for k in keys]); rec.append((str(keys),eval(ft)));rec.append((str(keys)+'diag',eval(np.c_[ft,*[coords[k]*coords[l] for k,l in combinations_with_replacement(keys,2)]])))
 display(rec)
