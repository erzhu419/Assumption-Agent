import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; T=-np.expm1(-t[:,None]*[1.8,.4]); W=np.c_[y,z@np.linalg.pinv(T.T)]
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(u-v,axis=1)<1; b,d=np.array([np.linalg.lstsq(np.c_[np.ones(sum(keep)),u[keep,i]],v[keep,i],rcond=None)[0] for i in range(4)]).T
funcs={'cube':lambda x:x**3,'ratcube':lambda x:x**3/(1+.22*x*x),'ratlin':lambda x:x/(1+.22*x*x),'satcube':lambda x:np.tanh(x**3),'tanhrat':lambda x:np.tanh(x**3/(1+.22*x*x)),'sqrtcube':lambda x:x**3/np.sqrt(1+.22*x**6),'atan':lambda x:np.arctan(x**3),'sin':lambda x:np.sin(x**3),'pow':lambda x:x**3/(1+.22*x**6),'abs':lambda x:x*abs(x),'sinlin':lambda x:np.sin(1.25*x),'tanhlin':lambda x:np.tanh(1.25*x)}
for name,X,Y,n in [('raw',x,W,52),('inverse',(x-b)/d,W,52),('forward',x*d+b,W,52)]+[(k+'_'+j,q['u'],vv,19) for k,q in s.items() if k in ('panel_b','panel_c') for j,vv in q.items() if j!='u']:
 print(name,flush=True); R=[]
 for fn,f in funcs.items():
  for base in range(5 if n==52 else 3):
   A=np.c_[np.ones(len(X)),f(X)];
   if base>=1:A=np.c_[A,X]
   if base>=2:A=np.c_[A,X**2]
   if base>=3:A=np.c_[A,*[np.prod(X[:,i],axis=1) for i in combinations(range(4),base-1)]]
   cf=np.linalg.lstsq(A[:n],Y[:n],rcond=None)[0]; P=A@cf; V=np.c_[P[:,0],P[:,1:]@T.T] if n==52 else P; true=np.c_[y,z] if n==52 else Y
   tr=np.mean(abs(V[:n]-true[:n]),axis=0); te=np.mean(abs(V[n:]-true[n:]),axis=0); R.append((np.mean(te),fn,base,tr,te))
 for r in sorted(R,key=lambda r:r[0])[:12]: print(r[1:3], 'train', np.round(r[3],4), 'test',np.round(r[4],4))
