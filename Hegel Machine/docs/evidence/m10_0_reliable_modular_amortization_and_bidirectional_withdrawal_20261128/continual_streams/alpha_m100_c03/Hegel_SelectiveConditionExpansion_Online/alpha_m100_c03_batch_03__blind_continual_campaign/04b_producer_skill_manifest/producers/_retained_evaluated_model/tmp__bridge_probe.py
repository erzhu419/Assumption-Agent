import numpy as np,pickle
from itertools import combinations
from scipy.linalg import lstsq
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']
def design(X,kind):
 a=[np.ones(len(X)),*X.T]
 if kind in ('diag2','full2','diag3','full3','sin','tanh','rational','sin2'): a.extend((X**2).T)
 if kind in ('full2','full3','sin','tanh','rational','sin2'): a.extend(X[:,i]*X[:,j] for i,j in combinations(range(4),2))
 if kind in ('diag3','full3'): a.extend((X**3).T)
 if kind=='full3': a.extend(X[:,i]**2*X[:,j] for i in range(4) for j in range(4) if i!=j); a.extend(X[:,i]*X[:,j]*X[:,k] for i,j,k in combinations(range(4),3))
 if kind=='sin': a.extend(np.sin(X).T)
 if kind=='sin2': a.extend(np.sin(2*X).T)
 if kind=='tanh': a.extend(np.tanh(X).T)
 if kind=='rational': a.extend((X/(1+X*X)).T)
 return np.column_stack(a)
def fit(A,Y):
 w=np.ones(len(Y)); b=lstsq(A,Y)[0]
 for _ in range(35):
  e=np.linalg.norm(A@b-Y,axis=1); s=max(np.median(e),1e-6); w=1/np.sqrt(1+(e/(2*s))**2); b=lstsq(A*w[:,None],Y*w[:,None])[0]
 return b
if __name__=='__main__':
 print('rows',np.column_stack([X,Y])[::3].round(4)); S=C['single_channel_sweeps']; print('sweep',np.column_stack(list(S.values())).round(4)); print('coverage', np.min(X,axis=0),np.max(X,axis=0))
 A=design(X,'full2'); b=fit(A,Y); out=np.linalg.norm(A@b-Y,axis=1)>0.6; print('spikes',np.flatnonzero(out), 'coeffs full2',b.round(4))
 rng=np.random.default_rng(19); folds=[*np.array_split(rng.permutation(len(X)),5),np.argsort(np.linalg.norm(X,axis=1))[-16:],np.where(np.count_nonzero(X,axis=1)<=1)[0]]
 for k in ['linear','diag2','full2','diag3','full3','sin','tanh','rational','sin2']:
  A=design(X,k); print(k,'rank',np.linalg.matrix_rank(A),'by fold med / p90: ',end='')
  for te in folds:
   tr=np.setdiff1d(np.arange(len(X)),te); e=np.linalg.norm(A[te]@fit(A[tr],Y[tr])-Y[te],axis=1)[~out[te]]; print(np.percentile(e,[50,90]).round(5),end=' ')
  print()
