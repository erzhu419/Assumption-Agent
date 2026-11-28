import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m
from panels14 import S,ls,aff
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.spatial.distance import cdist
tr=S['train']; z=m._inverse(tr['x'],m._calibrate(S['panel_a'])); y=tr['y']; T=tr['traj']; Y=np.c_[y,T]; u,v=S['panel_b'].values(); all=np.arange(len(y)); folds=[all[52:]]+list(np.array_split(np.random.default_rng(14).permutation(len(y)),4)); ix=[np.setdiff1d(all,f) for f in folds]

def prim(x,p,k):
 a=abs(x); s=np.sign(x); d=a-p[0];
 if k==0: h=np.maximum(d,0)**p[1]
 if k==1: h=(.5*(d+np.sqrt(d*d+p[2]**2)))**p[1]
 if k==2: h=np.logaddexp(0,p[2]*d)**p[1]/p[2]**p[1]
 if k==3: h=a*expit(p[2]*d)
 if k==4: h=np.maximum(d,0)+p[1]*np.maximum(d,0)**2
 return s*h

def proj(D,v):
 a=D-D.mean(0); b=v-v.mean(0); R,sc=orthogonal_procrustes(a,b); w=R*sc/np.sum(a*a); return w,v.mean(0)-D.mean(0)@w

def opfit(k,opt,dd):
 def ds(x,p):
  a=prim(x,p,k)
  if opt==1: return aff(np.c_[a,x])
  if opt==2: return aff(np.c_[a,x,np.sin(p[-1]*x)])
  if opt==3: return a+p[-2]*np.sin(p[-1]*x)
  if opt==4: return a+p[-3]*x+p[-2]*np.sin(p[-1]*x)
  return a
 def predict(p,x):
  D=ds(u,p); E=ds(x,p)
  if opt in (1,2): return E@ls(D,v)
  w,b=proj(D,v); return E@w+b
 p0,lo,hi=[[.34,1.3,.05],[.01,.3,.0001],[1.0,3,150]]
 if k==2:p0[2]=25;lo[2]=1
 if k==3:p0[2]=10;lo[2]=1
 if opt==4:p0+=[0];lo+=[-3];hi+=[3]
 if opt>=3:p0+=[.005];lo+=[-.2];hi+=[.2]
 if opt>=2:p0+=[dd];lo+=[.5];hi+=[50]
 fit=least_squares(lambda p:(predict(p,u)-v).ravel(),p0,bounds=(lo,hi),max_nfev=95)
 return np.std(fit.fun),(k,opt,fit.x),predict(fit.x,z)
ops=[(0,('base',),m._operator(S['panel_b'],z))]
for k in range(5):
 for opt in range(5):
  for d in ([4,9,15,22,34] if opt>=2 else [0]):ops.append(opfit(k,opt,d))
ops=ops[:1]+sorted(ops[1:],key=lambda a:a[0])[:17]; print('ops',[(a,b) for a,b,_ in ops],flush=True)
np.savez('/tmp/lastops14.npz',**{f'b{i}':q[-1] for i,q in enumerate(ops)})
print('T SVD',np.linalg.svd(T[:52],compute_uv=False),'yT',[(i,np.mean(abs(aff(T[52:,:i])@ls(aff(T[:52,:i]),y[:52])-y[52:]))) for i in (1,2,3,8)],flush=True)
def design(z,b,mode):
 if mode=='full':return m._quadratic(np.c_[z,b])
 if mode=='sep':return np.c_[m._quadratic(z),m._quadratic(b)[:,1:],z*b,z*np.roll(b,-1,axis=1),b*np.roll(z,-1,axis=1)]
 if mode=='zexpand':return np.c_[m._quadratic(np.c_[z,b]),z**3,z**4,b**3]
 if mode=='res':
  h=b-aff(z)@ls(aff(z),b);return np.c_[m._quadratic(np.c_[z,h]),z**3,h**3,z*z*h,z*h*h]
 if mode=='raw':return np.c_[m._quadratic(np.c_[z,b]),(z*z)*b,z*b*b,z*z*np.roll(b,-1,axis=1),b*b*np.roll(z,-1,axis=1)]
 if mode=='interaction':return np.c_[m._quadratic(np.c_[z,b]),z*np.roll(z,-1,axis=1)*b,z*b*np.roll(b,-1,axis=1),z**3,b**3]

def fit(A,B,yt,l,q):
 scale=np.r_[1,np.maximum(A[:,1:].std(0),.035)]; A=A/scale; B=B/scale; p=np.r_[0,np.repeat(l,A.shape[1]-1)]; H=A.T@A; h=A.T@yt; w=np.linalg.solve(H+np.diag(p+1e-10),h)
 for _ in range(8 if q else 0):
  pp=p*(.06/(abs(w)+.004))**q; w=np.linalg.solve(H+np.diag(pp+1e-10),h)
 return B@w
result=[]; preds=[]
for i,(_,_,b) in enumerate(ops):
 for mode in ['full','sep','zexpand','res','raw','interaction']:
  a=design(z,b,mode)
  for l in [.0005,.003,.012,.045,.16,.58]:
   for q in [0,.65,1.45]:
    ps=[fit(a[j],a[f],y[j],l,q) for j,f in zip(ix,folds)]; es=np.array([abs(p-y[f]).mean() for p,f in zip(ps,folds)]);result.append((es[0],es[1:].max(),es[1:].mean(),i,mode,l,q));preds.append(ps)
print('rank',*sorted(result)[:23],sep='\n');print('rob',*sorted(result,key=lambda a:np.mean(a[:3]))[:12],sep='\n')
for k in [3,10,25,60,160]:
 for sort in [lambda a:a[0],lambda a:np.mean(a[:3]),lambda a:np.mean(a[1:3])]:
  ids=sorted(range(len(result)),key=lambda j:sort(result[j]))[:k];print('ens',k,[round(abs(np.mean([preds[i][j] for i in ids],0)-y[f]).mean(),5) for j,f in enumerate(folds)], [result[i][3:] for i in ids[:5]])
