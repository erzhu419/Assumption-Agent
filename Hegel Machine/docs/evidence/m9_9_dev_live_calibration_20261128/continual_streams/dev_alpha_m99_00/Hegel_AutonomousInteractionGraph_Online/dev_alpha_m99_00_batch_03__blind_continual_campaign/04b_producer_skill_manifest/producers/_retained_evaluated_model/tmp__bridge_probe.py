import numpy as np,pickle,itertools
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; x=C['true_channel_loads']; y=C['bridge_raw']
np.set_printoptions(precision=5,suppress=True,linewidth=130)
print('axis rows:',np.column_stack([x[:16],y[:16]]));print('remaining patterns',np.unique(np.count_nonzero(x[16:],axis=1),return_counts=True))
print('sweep',[(k,np.shape(v)) for k,v in C['single_channel_sweeps'].items()]); print('top-level',list(D))
def fit(A,Y):
 b=np.linalg.lstsq(A,Y,rcond=None)[0]; w=np.ones(len(A));
 for _ in range(30):
  r=np.linalg.norm(A@b-Y,axis=1); w=np.minimum(1,max(np.median(r)*0.7,1.5e-4)/np.maximum(r,1e-12)); b=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
 return b
pairs=np.array(list(itertools.combinations(range(4),2))); i,j=pairs.T
base=np.c_[np.ones(len(x)),x]; cross=x[:,i]*x[:,j]
models={'affine':base,'xabs':np.c_[base,x*abs(x)],'cubic':np.c_[base,x**3],'full quadratic':np.c_[base,x*x,cross], 'quad+cubic':np.c_[base,x*x,cross,x**3], 'cross+xabs':np.c_[base,cross,x*abs(x)], 'cross+cubic':np.c_[base,cross,x**3]}
for a in (.25,.5,1,1.5,2,3):
 models['rational '+str(a)]=np.c_[base,cross,x**3/(1+a*x*x)]; models['sine '+str(a)]=np.c_[base,cross,np.sin(a*x)]
rng=np.random.default_rng(17); folds=np.array_split(rng.permutation(len(x)),5); tails=np.argsort(np.linalg.norm(x,axis=1))[-12:]
for k,A in models.items():
 b=fit(A,y); r=np.linalg.norm(A@b-y,axis=1); pred=np.empty_like(r)
 for te in folds:
  tr=np.setdiff1d(np.arange(len(x)),te); pred[te]=np.linalg.norm(A[te]@fit(A[tr],y[tr])-y[te],axis=1)
 tr=np.setdiff1d(np.arange(len(x)),tails); t=np.linalg.norm(A[tails]@fit(A[tr],y[tr])-y[tails],axis=1)
 print(k,'in sample',np.quantile(r,[.5,.85,.9]),'OOF',np.quantile(pred,[.5,.85,.9]),'tail',np.quantile(t,[.5,.85]),'outliers',np.flatnonzero(r>.01) if np.median(r)<.004 else '')
 if k in ('cross+cubic','cross+xabs','sine 2'):print(b)
