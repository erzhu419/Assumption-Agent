import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']; N=len(X)

def basis(x,kind):
 cols=[np.ones(len(x)),*x.T]
 if kind!='affine': cols += [x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]
 if kind=='cubic': cols += [x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations_with_replacement(range(4),3)]
 if kind=='axis3': cols += list((x**3).T)
 if kind=='axis5': cols += list((x**3).T)+list((x**5).T)
 return np.column_stack(cols)

def fit(a,b):
 keep=np.ones(len(a),bool)
 for i in range(10):
  coeff=lstsq(a[keep],b[keep])[0]; r=np.linalg.norm(a@coeff-b,axis=1); k=r<max(0.015,4*np.median(r))
  if np.array_equal(k,keep): break
  keep=k
 return coeff

if __name__=='__main__':
 print('design groups',np.unique(np.count_nonzero(X,axis=1),return_counts=True));print('cal row 8..19\n',np.round(np.c_[X,Y][8:20],5)); print('cal last 8\n',np.round(np.c_[X,Y][-8:],5))
 folds=np.random.default_rng(8).permutation(N)%5
 edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 for kind in ['affine','quadratic','axis3','axis5','cubic']:
  a=basis(X,kind); full=fit(a,Y); out=np.linalg.norm(a@full-Y,axis=1)>2; pred=np.empty_like(Y)
  for f in range(5):
   m=folds!=f;pred[~m]=a[~m]@fit(a[m],Y[m])
  err=np.linalg.norm(pred-Y,axis=1); er=np.linalg.norm(a[edge]@fit(a[~edge],Y[~edge])-Y[edge],axis=1)
  print(kind,'spikes',np.flatnonzero(out),'OOF median/p90/max',np.quantile(err[~out],[.5,.9,1]),'edge clean p90/max',np.quantile(er[~out[edge]],[.9,1]));
  if kind=='axis5':print('coeff row0/linear/axis3/axis5\n',np.round(full[np.r_[0:5,15:23]],5))
 print('nearest rows', np.quantile(np.partition(cdist(X,X),1,axis=1)[:,1],[0,.5,1]))
