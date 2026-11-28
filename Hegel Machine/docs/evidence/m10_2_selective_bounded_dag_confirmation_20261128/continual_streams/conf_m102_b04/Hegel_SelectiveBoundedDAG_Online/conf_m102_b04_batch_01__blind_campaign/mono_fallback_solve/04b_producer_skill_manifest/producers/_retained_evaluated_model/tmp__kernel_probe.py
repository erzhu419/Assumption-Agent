from operators import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
x=s['train']['x'];z=s['train']['traj'];y=s['train']['y']; p=s['panel_c'];u=p['u'];v=p['v1'];w=p['v2']; b=fit(basis(s['panel_b']['u'],1),s['panel_b']['v']);c=np.load('/tmp/cal.npy');
def cal(x):return (x-c[0])/c[1:].diagonal()
def rot(x):return basis(x,1)@b
def dv(x):return np.c_[np.ones(len(x)),np.arctan(2.12*x)]
def dw(x):return np.c_[basis(x,1),x*x/(1+.1616*x*x)]
cv=fit(dv(u),v);cw=fit(dw(u),w)
Q={}
for n,q in [('x',x),('c',cal(x)),('b',rot(x)),('bc',rot(cal(x))),('cb',cal(rot(x)))]:
 Q[n]=q
 for nm,q2 in [('V',dv(q)@cv),('W',dw(q)@cw),('a',np.arctan(2.12*q)),('s',q/np.sqrt(1+.22*q*q)),('t',np.tanh(1.25*q)),('r',q/(1+.22*q*q))]:
  Q[n+nm]=q2
  if nm in 'VW': Q[n+nm+'b']=rot(q2);Q[n+'+'+nm]=np.c_[q,q2]
def gp(q,Y,it,iv,k):
 ym=Y[it].mean(); sd=Y[it].std(); yy=(Y[it]-ym)/sd
 D=(q[:,None,:]-q[None,it,:])**2
 def K(th):
  W=np.exp(th[:q.shape[1]]); d=D@W
  if k=='rbf':return np.exp(-.5*d)
  if k=='rq':return 1/(1+.5*d)
  if k=='an':return np.exp(-.5*d)+np.exp(th[-2])*np.exp(-.5*D*W).mean(2)
 def obj(th,ret=False):
  k0=K(th); a=k0[it]+np.eye(len(it))*np.exp(th[-1]); ch=cho_factor(a); al=cho_solve(ch,yy)
  if ret:return (k0[iv]@al)*sd+ym
  return .5*yy@al+np.log(np.diag(ch[0])).sum()
 r=minimize(obj,np.r_[np.zeros(q.shape[1]),-2.,-9.] if k=='an' else np.r_[np.zeros(q.shape[1]),-9.],bounds=[(-11,7)]*q.shape[1]+([(-10,5)] if k=='an' else [])+[(-15,-2)],method='L-BFGS-B');return obj(r.x,True),r.fun
np.set_printoptions(precision=5,suppress=True)
print('corr y trajectory',np.corrcoef(np.c_[y,z].T)[0]);print('trajectory SV',np.linalg.svd(z,compute_uv=False)); print('y vs nonlinear trajectory regression',[(d,np.std(y-basis(z[:,-1:],d)@fit(basis(z[:,-1:],d),y))) for d in range(1,5)])
for j in range(2):
 R=[]
 for name,q in Q.items():
  if name.startswith('cb'):continue
  for k in ('rbf','rq','an'):
   E=[]; N=[]
   for hold in (np.arange(52,72),np.arange(0,20),np.arange(24,44),np.argsort(np.linalg.norm(x,axis=1))[-20:]):
    tr=np.setdiff1d(np.arange(len(x)),hold); V=np.linalg.svd(z[tr],full_matrices=False)[2][:1]; amp=z@V[0]; tar=(y,amp)[j]; pred,nll=gp(q,tar,tr,hold,k);E.append(np.mean(abs(pred-tar[hold])));N.append(nll)
   R.append((E,name,k,np.mean(N)))
 print('target',j)
 for key in (lambda r:np.mean(r[0]),lambda r:r[0][0], lambda r:r[-1]):
  print('ranking');
  for r in sorted(R,key=key)[:8]:print(np.array(r[0]),r[1:])
