import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; bias=np.array([-.06,.07,.06,-.05]); gain=np.array([.8,1.28,1.3,.79]); clean=np.max(abs(y-bias),axis=1)<5; x=x[clean]; y=y[clean]; ax=np.count_nonzero(x,axis=1)==1; n=len(x)
R=(y-bias)/gain-x-.25*np.sin(2.5*x); X=np.stack([np.roll(x,-i,axis=1) for i in range(4)],axis=1); print('shared opposite axis',np.c_[X[ax,:,2].ravel(),R[ax].ravel()][X[ax,:,2].ravel()!=0]);
for d in [2,3,4,5,6]:
 I=[i for k in range(1,d+1) for i in combinations_with_replacement(range(4),k) if len(set(i))>1 or i[0]==2]; A=np.stack([X[:,:,i].prod(2) for i in I],axis=-1); ids=np.arange(n); folds=np.array_split(np.random.default_rng(0).permutation(ids[~ax]),5); edge=ids[~ax & (abs(x).max(1)>1.35)];
 def fit(val):
  tr=~np.isin(ids,val); p=np.linalg.lstsq(A[tr].reshape(-1,len(I)),R[tr].ravel(),rcond=1e-8)[0]; e=(A@p-R); return p,np.sqrt(np.mean(e[val]**2)),np.max(abs(e[val])),np.sqrt(np.mean(e[tr]**2))
 p,*_=fit([]) if False else (np.linalg.lstsq(A.reshape(-1,len(I)),R.ravel(),rcond=1e-8)[0],); print('shared poly',d,'random worst',np.max([fit(v)[1:] for v in folds],axis=0),'edge',fit(edge)[1:]); print('top',[(I[i],round(p[i],6)) for i in np.argsort(-abs(p))[:14]])
U=X[ax,:,2].ravel(); V=R[ax].ravel(); v=U!=0; u=U[v]; z=V[v]; fns={'lin':lambda x:x,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x)}; ev={'x2':lambda x:x*x,'rat2':lambda x:x*x/(1+x*x),'tanh2':lambda x:np.tanh(x)**2,'cos':lambda x:1-np.cos(x),'exp2':lambda x:1-np.exp(-x*x)}
q=[]
for a,f in fns.items():
 for b,g in ev.items():
  fun=lambda p: p[0]*f(p[1]*u)+p[2]*g(p[3]*u)-z
  k=least_squares(fun,[-.25,1.,.1,1.],bounds=([-4,.05,-4,.05],[4,4,4,4]),max_nfev=250); q.append((np.max(abs(k.fun)),a,b,k.x))
print('axis fits',sorted(q,key=lambda r:r[0])[:12]); print('batch study count',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
