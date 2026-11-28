import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(x)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(z,d):return np.column_stack([np.prod(z[:,i],axis=1) if i else np.ones(len(z)) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; A=aff(u); g=np.ones(len(u),bool)
for i in range(4):
 c=fit(A[g],v[g]); e=np.linalg.norm(v-A@c,axis=1);g=e<4*np.median(e)
a=(x-c[0])/np.diag(c[1:]); b=s['panel_b']; B=fit(aff(b['u']),b['v']); z=aff(a)@B
np.set_printoptions(precision=4,suppress=True)
def metrics(D,target=y):
 c=fit(D[:52],target[:52]); e=D@c-target
 return np.mean(abs(e[:52])),np.mean(abs(e[52:])),np.max(abs(e[52:])),c
from itertools import product
from scipy.optimize import least_squares
w=z; g,h,j,k=w.T; S=g+h-j-k; P=g*j+h*k; F=aff(w); freq=s['telemetry_vec'][0]
folds=[np.arange(52,72),np.arange(56,72),*np.array_split(np.random.default_rng(813).permutation(n),5)]; splits=[(np.setdiff1d(np.arange(n),v),v) for v in folds]
cs=[]
for name,V in [('signed',S),('total',g+h+j+k),('alt',g-h+j-k),('alt2',g-h-j+k),('pair',P),('pairdiff',g*j-h*k),('left',g+h),('right',j+k),('norm',np.sqrt(np.sum(w*w,axis=1)))]:
 for fun in ['sin','tanh','both','rat']:
  G={'sin':np.sin(freq*S)[:,None],'tanh':np.tanh(S)[:,None],'both':np.c_[np.sin(freq*S),np.tanh(S)],'rat':(S/(1+S*S))[:,None]}[fun]
  for base in ['lin','quad','pair']:
   A={'lin':F,'quad':poly(w,2),'pair':np.c_[F,P]}[base]
   for lam in [.1,.2,.5,1,1.7]:
    D=np.c_[A,G,P/(1+lam*V**2)]; es=[np.mean(abs(D[v]@fit(D[u],y[u])-y[v])) for u,v in splits]; c0=fit(D,y); cs.append((np.mean(es),max(es),es[:2],name,fun,base,lam,np.std(y-D@c0),c0))
for row in sorted(cs,key=lambda r:r[0])[:12]: print(row)
print('FREE SIGNED GATE')
for kind in ['sin','tanh','both']:
 for free in [False,True]:
  ps=[]; es=[]
  for u,v in splits+[(np.arange(n),np.arange(n))]:
   def design(p):
    v=S if not free else S+w@p[1:5]+p[5]; T={'sin':np.sin(freq*S)[:,None],'tanh':np.tanh(S)[:,None],'both':np.c_[np.sin(freq*S),np.tanh(S)]}[kind]; return np.c_[F,T,P/(1+np.exp(p[0])*v*v)]
   def res(p):
    D=design(p)[u]; return D@fit(D,y[u])-y[u]
   q=least_squares(res,np.r_[np.log(.2),np.zeros(5 if free else 0)],max_nfev=150).x; D=design(q); C=fit(D[u],y[u]); ps.append(q); es.append(np.mean(abs(D[v]@C-y[v])))
  print(kind,free,np.round(es,6),'lam',np.exp(ps[-1][0]),'gate',ps[-1][1:],'coef',C)
print('trajectory affine coeffs',fit(F,t)); print('trajectory per horizon rms',np.std(t-F@fit(F,t),axis=0)); print('label y diff from summary',y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std())
