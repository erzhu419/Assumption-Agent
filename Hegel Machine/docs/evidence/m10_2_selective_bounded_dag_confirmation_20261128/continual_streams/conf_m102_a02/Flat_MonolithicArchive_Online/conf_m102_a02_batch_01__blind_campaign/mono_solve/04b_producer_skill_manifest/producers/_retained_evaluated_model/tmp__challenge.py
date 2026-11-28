import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibration,_time_basis
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; b,d=_calibration(s['panel_a']); X=(x-b)/d
Y=np.c_[y,z]; folds=[np.arange(72)]+[np.random.default_rng(j).permutation(72) for j in range(1,12)]
i,j=np.triu_indices(4,1); l,m,n=np.array(list(combinations(range(4),3))).T
funs={'tanh':lambda x:np.tanh(1.25*x),'sin':lambda x:np.sin(2.9*x),'rat':lambda x:x/(1+.75*x*x),'tanh3':lambda x:np.tanh(1.25*x)**3,'warp':lambda x:x+.22*x**3}

def des(X,f,extra):
 S=f(X); A=np.c_[np.ones(len(X)),X,S]; opts=[X**2,X[:,i]*X[:,j],X[:,l]*X[:,m]*X[:,n],(X[:,i]*S[:,j]+S[:,i]*X[:,j])/2,(X[:,i]*S[:,j]-S[:,i]*X[:,j])/2,S[:,i]*S[:,j],X*S];
 return A,np.column_stack([opts[k] for k in extra]) if extra else np.zeros((len(X),0))

def cv(A,E,ly,lz):
 out=[]
 for q in folds:
  a,c=q[:52],q[52:]; B=_time_basis(t,z[a]); amp=z[a]@np.linalg.pinv(B.T); vals=[]
  for w,pen in [(y[a,None],ly),(amp,lz)]:
   D=np.c_[A,E/(pen**.5)]; R=np.diag(np.r_[[0.]*A.shape[1],[1.]*E.shape[1]]); vals.append(D[c]@np.linalg.lstsq(np.r_[D[a],R],np.r_[w,np.zeros((len(R),w.shape[1]))],rcond=None)[0])
  r=abs(np.c_[vals[0],vals[1]@B.T]-Y[c]); ext=(abs(X[c]).max(1)>=1.); out.append([r[:,0].mean(),r[ext,0].mean(),r[:,1:].mean(),r[:,1:].mean(0).max()]);
 return np.array(out)
R=[]
for label,f in funs.items():
 for extra in [(),(0,),(1,),(0,1),(1,2),(0,1,2),(0,1,3,4,5,6),(0,1,2,3,4,5,6)]:
  A,E=des(X,f,extra)
  for ly,lz in ([(1,1)] if not extra else [(.03,.03),(.25,.25),(1,1),(.1,1)]):
   v=cv(A,E,ly,lz); R.append((v.mean(0)[0]+v.max(0)[1]+v.mean(0)[2],label,extra,ly,lz,v))
for score,k,e,ly,lz,v in sorted(R,key=lambda r:r[0])[:20]:print(k,e,ly,lz,'split',v[0].round(4),'mean',v.mean(0).round(4),'max',v.max(0).round(4))
for k in funs:
 v=next(r[-1] for r in R if r[1:3]==(k,()));print(k,'base',v[0].round(4),v.mean(0).round(4),v.max(0).round(4))
