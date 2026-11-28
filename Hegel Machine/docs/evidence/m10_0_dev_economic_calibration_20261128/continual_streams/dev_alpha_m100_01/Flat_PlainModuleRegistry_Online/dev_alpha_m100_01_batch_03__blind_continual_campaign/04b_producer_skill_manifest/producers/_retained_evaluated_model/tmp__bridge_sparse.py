import numpy as np,pickle
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps'];
def design(x,j,kind='cyclic'):
 a,b=x[:,j],x[:,(j+1)%4]; terms=[np.ones(len(x)),*x.T,a*b,a**3];
 if kind=='cubic': terms=[np.ones(len(x))]+[np.prod(x[:,ii],axis=1) for p in (1,2,3) for ii in combinations_with_replacement(range(4),p)]
 return np.column_stack(terms)
def fit(A,y):
 w=np.ones(len(y));
 for _ in range(30):
  t=w**.5; b=np.linalg.lstsq(t[:,None]*A,t*y,rcond=None)[0]; e=A@b-y; s=1.4826*np.median(abs(e-np.median(e)));w=1/(1+(e/max(2.5*s,1e-9))**2)
 return b
folds=[np.array_split(np.random.default_rng(31).permutation(len(X)),5),[np.arange(i,i+4) for i in range(0,16,4)]]
for kind in ('cyclic','cubic'):
 for fs in folds:
  es=[]
  for ts in fs:
   tr=np.setdiff1d(np.arange(len(X)),ts); A=[design(X,j,kind) for j in range(4)];pred=np.column_stack([A[j][ts]@fit(A[j][tr],Y[tr,j]) for j in range(4)]);e=np.linalg.norm(pred-Y[ts],axis=1);es.extend(e[~np.isin(ts,[13,14,15])]);
  print(kind,len(es),'q50/90/max',np.quantile(es,[.5,.9,1]))
for j in range(4):
 b=fit(design(X,j),Y[:,j]); use=S['lab_id']==j; q=np.zeros((use.sum(),4));q[:,j]=S['true_stress'][use];e=design(q,j)@b-S['cmd_stress'][use];print('channel',j,'coeff',np.round(b,6),'sweep residual',np.round(e,6))
