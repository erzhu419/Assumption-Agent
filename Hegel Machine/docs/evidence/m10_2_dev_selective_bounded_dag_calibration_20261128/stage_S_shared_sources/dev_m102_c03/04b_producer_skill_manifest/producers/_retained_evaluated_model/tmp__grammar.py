import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement,product
from sklearn.linear_model import lars_path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
p=s['panel_a']; u=bias(p['u']); v=p['v']; good=np.linalg.norm(v-u@fit(u,v),axis=1)<3; A=fit(u[good],v[good]); a=lambda x:(x-A[0])/np.diag(A[1:]); ai=lambda x:bias(x)@A
p=s['panel_b']; B=fit(bias(p['u']),p['v']); b=lambda x:bias(x)@B; bi=lambda x:(x-B[0])@np.linalg.inv(B[1:]);
p=s['panel_c'];u=p['u'];
cs=[]
for k,phi,q0 in [('v1',lambda z,q:z/(1+q*abs(z)),.84),('v2',lambda z,q:np.expm1(-q*z*z),.295)]:
 v=p[k]; F=lambda z,q:np.c_[bias(z),phi(z,q[0])]; o=least_squares(lambda q:(F(u,q)@fit(F(u,q),v)-v).ravel(),[q0]); W=fit(F(u,o.x),v);print(k,o.x,'error',np.abs(o.fun).mean(),'ratios',np.round(W[1:5]/W[5:],3));cs.append(lambda z,F=F,q=o.x,W=W:F(z,q)@W)
x=np.r_[s['train']['x'],s['test']['x']];tr=s['train']['traj'];t=s['eval_times'];Q=-np.expm1(-t[:,None]*[.392,1.857]);Y=np.c_[s['train']['y'],fit(Q,tr.T).T];
def report(P,Y,idx):return np.array([np.abs(P[:,0]-Y[:,0]).mean(),np.abs(P[:,1:]@Q.T-tr[idx]).mean(),np.abs((P[:,1:]-Y[:,1:])@Q[-1]).mean()])
# Compare low-dimensional coordinate grammars before screening their joint dictionary.
trans={'x':x,'a':a(x),'A':ai(x),'b':b(x),'ba':b(a(x)),'bA':b(ai(x)),'B':bi(x),'Ba':bi(a(x)),'BA':bi(ai(x))}
for k,z in list(trans.items()):
 for j,c in enumerate(cs):trans[k+str(j)]=c(z)
rows=[]
for key,z in trans.items():
 for typ in ['rational','soft','trig','poly','mixed','cpair','quotient']:
  F=[bias(z)]
  if typ=='rational':F+=[z/(1+q*z*z) for q in [.35,1,2]];F+=[z*z/(1+q*z*z) for q in [.35,1]]
  if typ=='soft':F+=[z/(1+q*abs(z)) for q in [.42,.84,1.68]];F+=[abs(z),z*z]
  if typ=='trig':F+=[f(q*z) for q in [.8,1.6,2.4] for f in [np.sin,np.cos]]
  if typ=='poly':F+=[z**k for k in [2,3,4,5,6]]
  if typ in ['mixed','cpair','quotient']:
   F+=[z*z,z**3,z**4]
   for i,j in combinations(range(4),2):
    u,v=z[:,i:i+1],z[:,j:j+1];F+=[u*v]
    if typ=='mixed':F+=[u*v*v,u*u*v]
    if typ=='cpair':F+=[f(q*(u+sg*v)) for f in [np.sin,np.cos,np.tanh] for q in [1,2] for sg in [-1,1]]
    if typ=='quotient':F+=[u/(1+.8*v*v),v/(1+.8*u*u),u*v/(1+.8*(u*u+v*v))]
  F=np.column_stack(F); X=F[:52]; G=X.T@X; scale=np.maximum(np.sqrt(np.diag(G)),1e-6); F=F/scale; X=F[:52]; G=X.T@X
  for reg in [1e-5,.001,.01]:
   if X.shape[1]>52:
    inv=np.linalg.inv(X@X.T+reg*np.eye(len(X))); W=X.T@inv@Y[:52]; loo=Y[:52]-(inv@Y[:52])/np.diag(inv)[:,None]
   else:
    Xinv=np.linalg.solve(G+reg*np.eye(len(G)),X.T);W=Xinv@Y[:52];loo=Y[:52]-(Y[:52]-X@W)/(1-np.sum(X*Xinv.T,1))[:,None]
   rows.append((report(loo,Y[:52],slice(0,52)),report(F[52:72]@W,Y[52:],slice(52,72)),key,typ,reg))
for j in [0,1]:
 print('best loo',j);print(*sorted(rows,key=lambda r:r[0][j])[:9],sep='\n'); print('diagnostic best tail',j); print(*sorted(rows,key=lambda r:r[1][j])[:5],sep='\n')
