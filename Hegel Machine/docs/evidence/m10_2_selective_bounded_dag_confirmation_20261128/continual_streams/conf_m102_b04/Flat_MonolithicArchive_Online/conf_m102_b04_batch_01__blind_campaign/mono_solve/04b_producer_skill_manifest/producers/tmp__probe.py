import pickle,numpy as np
from itertools import combinations_with_replacement as comb
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def poly(x,d): return np.column_stack([np.ones(len(x))]+[x[:,j].prod(1) for k in range(1,d+1) for j in comb(range(x.shape[1]),k)])
def fit(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
D={}; print('eval_times',s['eval_times'],'telemetry',s['telemetry_vec']);
for a in 'abc':
 d=s['panel_'+a]; u=d['u']; X=poly(u,1)
 for k,w in d.items():
  if k=='u': continue
  C=fit(X,w)
  if a=='a':
   for i in range(10):
    ix=np.argsort(np.linalg.norm(X@C-w,axis=1))[:39]; C=fit(X[ix],w[ix]);
  print(a,k,'coeff',C,'row residuals',np.sort(np.linalg.norm(X@C-w,axis=1))); D[a+k]=C
  if a=='a':continue
  for fn in (lambda t:t, np.tanh,np.sin,lambda t:t/np.sqrt(1+t*t),lambda t:t/(1+t*t)):
   out=[]
   for j in range(4):
    y=w[:,j]; c=C[:,j]; best=min([least_squares(lambda p:p[0]+p[1]*fn(X@p[2:])-y,np.r_[0.,g,c/g],max_nfev=400) for g in [.65,1.3,3]],key=lambda o:np.linalg.norm(o.fun)); out.append(np.std(best.fun));
   print(fn.__name__,out)
  for dd in [1,2,3]:
   F=poly(u,dd); out=[]
   for i in range(len(u)):
    m=np.arange(len(u))!=i; out.append(F[i]@fit(F[m],w[m])-w[i]);
   print('panel LOOCV polynomial',dd,np.sqrt(np.mean(np.array(out)**2,axis=0)))
x=s['train']['x']; y=s['train']['y']; Q=s['train']['traj']; U,sv,V=np.linalg.svd(Q,full_matrices=False); Y=np.c_[y,Q@V[0]]; print('time rank',sv,'profile',V[:2]);
A=D['av']; Vs={'raw':x,'fA':poly(x,1)@A,'iA':(x-A[0])@np.linalg.inv(A[1:]),'diagA':(x-A[0])/A[1:].diagonal()};
for k,z in list(Vs.items()):
 for l,C in D.items():
  if l!='av': Vs[k+'_'+l]=poly(z,1)@C
res=[]
for k,z in Vs.items():
 for g in [0,.04,.09,.2,.5,1,2]:
  for n in [1,2,3,4]:
   F=poly(z/np.sqrt(1+g*z*z),n); F=F/np.maximum(np.std(F[:52],axis=0),.15); F[:,0]=1
   for a in [1e-4,.003,.03,.15,.5]:
    C=np.linalg.solve(F[:52].T@F[:52]+a*np.diag(np.r_[0,np.ones(F.shape[1]-1)]),F[:52].T@Y[:52]); pred=F@C; e=np.abs(pred[52:]-Y[52:]).mean(0); tr=np.std(pred[:52]-Y[:52],axis=0); res.append([*e,k,g,n,a,*tr]);
for j in [0,1]: print('top target',j,*sorted(res,key=lambda r:r[j])[:16],sep='\n')
