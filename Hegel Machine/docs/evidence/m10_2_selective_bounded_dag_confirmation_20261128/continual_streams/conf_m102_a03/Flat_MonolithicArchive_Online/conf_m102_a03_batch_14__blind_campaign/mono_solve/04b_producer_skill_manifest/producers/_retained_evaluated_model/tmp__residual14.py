import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; n=len(y); S=np.load('/tmp/spaces.npz'); N=np.load('/tmp/newspaces.npz'); z=S['z'][:n]; b=N['thresholdFalse'][:n]; pair=z*np.roll(b,1,axis=1); p=pair.sum(1); af=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]; H=af(np.c_[b,p]); rg=np.random.default_rng(34); folds=[(np.arange(52),np.arange(52,72))]+[(i[:52],i[52:]) for i in [rg.permutation(n) for j in range(7)]]
def score(A):
 er=np.array([np.mean(abs(A[v]@fit(A[t],y[t])-y[v])) for t,v in folds]); return tuple(np.round([er[0],er.mean(),er.max(),np.mean(abs(A[:52]@fit(A[:52],y[:52])-y[:52]))],6))
D={'z':z,'b':b,'pair':pair,'C':S['C0'][:n],'D':S['D3'][:n],'p':p[:,None]}; R=[]; tests=[]
for k,a in D.items():
 tests.append((k,a));
 for j,c in D.items():
  if j<k:continue
  for name,A in [('outer',(a[:,:,None]*c[:,None,:]).reshape(n,-1)),('same',a*c) if a.shape[1]==c.shape[1] or min(a.shape[1],c.shape[1])==1 else ('same',a)]:
   tests.append((k+j+name,A))
   for i in range(A.shape[1]): tests.append((k+j+name+str(i),A[:,i,None]))
 for j in [1,2,3]:
  g=np.roll(a,j,axis=1)
  for sign in [-1,1]:tests.append((k+str(j)+str(sign),a+sign*g))
for k,A in tests.copy():
 if A.shape[1]>4:continue
 for name,f in [('abs',abs),('cube',lambda x:x**3),('tan',lambda x:np.tanh(2*x)),('sin',lambda x:np.sin(2*x)),('cos',lambda x:np.cos(2*x)),('rat',lambda x:x/(1+2*x*x))]:tests.append((k+name,f(A)))
for k,A in tests:
 for suffix,U in [('',H),(' with lin z',np.c_[H,z])]:
  W=np.c_[U,A]; R.append((score(W),k+suffix))
print('BASE',score(H)); print('MEAN',sorted(R,key=lambda x:x[0][1])[:25]); print('SPLIT',sorted(R)[:12]);
# Test if the residual is a nonlinear coupling involving the auxiliary panels.
for k in ['b','z','C','D','pair']:
 A=D[k];
 for j in ['p','b','z','C','D']:
  B=D[j];
  for family in ['sin','tanh','soft','cos','rat','softplus']:
   fn={'sin':np.sin,'tanh':np.tanh,'soft':lambda x:x/(1+abs(x)),'cos':np.cos,'rat':lambda x:x/(1+x*x),'softplus':lambda x:np.logaddexp(0,x)}[family]
   def design(q):return np.c_[H,B*fn(A@q[:-1]+q[-1])[:,None]]
   def solve(t,v):
    def res(q): W=design(q)[t]; return W@fit(W,y[t])-y[t]
    starts=[np.random.default_rng(i).normal(0,.7,A.shape[1]+1) for i in range(4)]; q=min((least_squares(res,x,bounds=(-6,6),max_nfev=90) for x in starts),key=lambda q:q.cost); W=design(q.x); return np.mean(abs(W[v]@fit(W[t],y[t])-y[v])),np.mean(abs(q.fun))
   err,tr=solve(*folds[0]);
   if err<.035 or tr<.015:print('nonlinear',k,j,family,err,tr,[round(solve(*f)[0],5) for f in folds[1:4]],flush=True)
