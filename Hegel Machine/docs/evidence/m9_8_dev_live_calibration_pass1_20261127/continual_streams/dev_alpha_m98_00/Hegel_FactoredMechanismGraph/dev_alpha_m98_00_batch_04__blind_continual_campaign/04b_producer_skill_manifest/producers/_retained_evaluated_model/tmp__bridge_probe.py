import pickle,itertools,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
def mat(x, mode):
 terms=[()]+[(j,) for j in range(4)]
 if mode != 'linear':
  terms+=list(itertools.combinations_with_replacement(range(4),2))
  terms+= [(j,j,j) for j in range(4)] if mode=='owncubic' else (list(itertools.combinations_with_replacement(range(4),3)) if mode=='cubic' else [])
 return np.array([x[:,p].prod(1) for p in terms]).T,terms

def fit(a,z):
 b=np.linalg.lstsq(a,z,rcond=None)[0]
 for s in (.8,.2,.06,.02,.006,.002,.0006):
  if len(a)<a.shape[1]+4: break
  for _ in range(16):
   w=(1+np.sum(((a@b-z)/s)**2,axis=1))**-.5
   b=np.linalg.lstsq(a*w[:,None],z*w[:,None],rcond=None)[0]
 return b

rng=np.random.default_rng(214); folds=list(np.array_split(rng.permutation(len(x)),5)); folds+= [np.arange(16),np.argsort(abs(x).max(1))[-12:]]
for mode in ('linear','quad','owncubic','cubic'):
 a,p=mat(x,mode); b=fit(a,y); e=np.linalg.norm(a@b-y,axis=1); print('\n',mode,'fit quantiles',np.round(np.quantile(e,[.25,.50,.85,.90,.95,1]),6),'gross rows',np.flatnonzero(e>1.)); clean=e<1
 for j in range(4):
  print('ch',j,[( ''.join(map(str,t)) or 'bias',round(v,5)) for t,v in zip(p,b[:,j]) if abs(v)>.001])
 err=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(len(x)),te); q=fit(a[tr],y[tr]); r=a[te]@q-y[te]; r=r[clean[te]]
  err.append(np.round([np.mean(r*r)**.5,np.max(abs(r))],6).tolist())
 print('CV clean RMS and max, 5 random folds then axes then extremes',err)
