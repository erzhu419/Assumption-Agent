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
A=models['quad+cubic']; B=fit(A,y)
r=np.linalg.norm(A@B-y,axis=1); clean=r<0.003
B=np.linalg.lstsq(A[clean],y[clean],rcond=None)[0]
print('feature order: intercept, linear 0:4, square 0:4, cross 01 02 03 12 13 23, cube 0:4');print(B)
print('clean max / min rejected',np.max(r[clean]), np.min(r[~clean]));print('mix bounds',x[16:].min(0),x[16:].max(0));print('singular values clean',np.linalg.svd(A[clean],compute_uv=False)[[0,-1]])
