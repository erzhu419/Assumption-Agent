import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; b=C['bridge_raw']; x=C['true_channel_loads']
rng=np.random.default_rng(602); fold=np.empty(len(x),int);fold[rng.permutation(len(x))]=np.arange(len(x))%5

def design(x,mode):
 a=[np.ones(len(x)),x]
 if mode in ('diag2','diag23','full2','full23'): a+=[x*x]
 if mode in ('diag3','diag23','full23'): a+=[x*x*x]
 if mode in ('full2','full23'): a+=[np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)])]
 if mode=='full23': a+=[np.column_stack([np.prod(x[:,ix],axis=1) for ix in combinations_with_replacement(range(4),3) if len(set(ix))>1])]
 return np.column_stack(a)

def fit(a,y):
 keep=np.ones(len(a),bool)
 for step in range(6):
  coef=np.linalg.lstsq(a[keep],y[keep],rcond=None)[0]; e=np.linalg.norm(a@coef-y,axis=1); new=e<max(0.001,5*np.median(e));
  if np.array_equal(new,keep) or new.sum()<a.shape[1]:break
  keep=new
 return coef
for mode in ('linear','diag2','diag3','diag23','full2','full23'):
 a=design(x,mode); pred=np.zeros_like(b)
 for f in range(5):
  j=fold!=f;pred[~j]=a[~j]@fit(a[j],b[j])
 e=np.linalg.norm(pred-b,axis=1); c=fit(a,b);print(mode,'oof quantiles',np.quantile(e,[.5,.9,.95,.96,1]),'outlying rows',np.where(e>.1)[0]);
 if mode in ('diag3','diag23','full23'):
  print('coefficients',np.round(c[:13],6),'tail max', abs(c[13:]).max(initial=0))
