from explore import *
from graph import panel,fw,bw
from sklearn.model_selection import KFold
from scipy.linalg import solve
from itertools import product
folds=list(KFold(5,shuffle=True,random_state=21).split(np.arange(52)))+[(np.arange(52),np.arange(52,72))]
G={'x':x,'z':z,'h':h,'Bz':fw(z),'Bx':fw(x),'ix':bw(x)}
for k,g in list(G.items()):
 a,b=panel(g); G.update({k+'c1':a,k+'c2':b,k+'cs':a+b,k+'cd':a-b,k+'cp':a*b})
def std(F,i):
 return F/np.maximum(np.sqrt(np.mean(F[i]**2,axis=0)),1e-8)
def err(F,pen,Y=R):
 E=[]
 for i,j in folds:
  A=std(F,i); Q=A[i]; M=Q.T@Q+np.diag(np.broadcast_to(pen,Q.shape[1])); p=A[j]@solve(M,Q.T@Y[i],assume_a='pos'); E.append(np.mean(abs(p-Y[j]),0))
 return np.array(E)
base=aff(z); out=[]
for k,g in G.items():
 for tag,h2 in [('id',g),('t.7',np.tanh(.7*g)),('t1.25',np.tanh(1.25*g)),('t2.2',np.tanh(2.2*g)),('sin1.25',np.sin(1.25*g)),('sin2.2',np.sin(2.2*g)),('r.22',g/(1+.22*g*g)),('r1.25',g/(1+1.25*g*g)),('plus',g+.22*np.tanh(1.25*g)),('minus',g-.22*np.tanh(1.25*g))]:
  A=np.c_[base,poly(h2,2)[:,1:]]
  for lam in [.005,.15,1.5]:
   E=err(A,np.r_[np.full(5,1e-6),np.full(A.shape[1]-5,lam)]);out.append((k,tag,lam,E[:-1].mean(0),E[-1]))
for j in range(3):
 print('SINGLE',j,*sorted(out,key=lambda t:t[3][j])[:12],sep='\n'); print('gate diagnostic',*sorted(out,key=lambda t:t[4][j])[:5],sep='\n')
# Sparse higher-order interactions and gated multivariate products.
out=[]
for k,g in list(G.items())[:6]:
 P=poly(g,4); F=poly(g,2); t1,t2=panel(g); th=np.tanh(1.25*g); si=np.sin(1.25*g); rr=1/(1+.22*g*g)
 mats=[('higher',P[:,15:]),('gate',np.c_[poly(th,2)[:,1:],poly(si,2)[:,1:],(F[:,:,None]*rr[:,None,:]).reshape(len(g),-1)]),('branch',np.c_[poly(t1,2)[:,1:],poly(t2,2)[:,1:],(t1[:,:,None]*t2[:,None,:]).reshape(len(g),-1)]),('pair',np.column_stack([fn(g[:,a]+sg*g[:,b]) for a in range(4) for b in range(a+1,4) for sg in [1,-1] for fn in [lambda v:np.tanh(1.25*v),lambda v:np.cos(1.25*v),lambda v:1/(1+.22*v*v)]]))]
 for name,M in mats:
  A=np.c_[F,M];
  for lam in [.03,.3,3,15,70]:
   E=err(A,np.r_[np.full(5,1e-6),np.full(10,.04),np.full(M.shape[1],lam)]);out.append((k,name,lam,E[:-1].mean(0),E[-1]))
for j in range(3):
 print('COMPOUND',j,*sorted(out,key=lambda t:t[3][j])[:10],sep='\n');print('gate diagnostic',*sorted(out,key=lambda t:t[4][j])[:5],sep='\n')
