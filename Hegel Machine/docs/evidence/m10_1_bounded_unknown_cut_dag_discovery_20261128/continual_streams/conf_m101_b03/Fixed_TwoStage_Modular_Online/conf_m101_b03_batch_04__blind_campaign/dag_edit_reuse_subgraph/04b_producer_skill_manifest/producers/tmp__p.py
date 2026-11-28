import numpy as np,pickle,sys
from itertools import combinations,combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=run_node({'__input__':s}); z=d['z']; x=z[:d['n_train']]; t=d['eval_times']; y=d['y']; T=d['traj']; Y=np.c_[y,T]; n=len(x)
ones=np.ones(n)
def poly(deg):return np.column_stack([ones]+[np.prod(x[:,c],axis=1) for k in range(1,deg+1) for c in combinations_with_replacement(range(4),k)])
def report(k,A):
 P=np.linalg.lstsq(A,Y,rcond=None)[0]; E=Y-A@P; H=(np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]**2).sum(1); res=np.abs(E/(1-H[:,None])); print(k,A.shape[1], 'train',np.mean(abs(E),0).round(4),'loo',res.mean(0).round(4),'tail',np.mean(abs(Y[-18:]-A[-18:]@np.linalg.lstsq(A[:-18],Y[:-18],rcond=None)[0]),0).round(4))
if __name__=='__main__':
 print('ranges',x.min(0),x.max(0),'t',t,'T svd',np.linalg.svd(T)[1],'T temporal modes',np.linalg.svd(T)[2][:4].round(4));print('T rows',T[:8].round(4)); print('x y',np.c_[x,y][:8].round(4))
 for k in (1,2,3):report('poly'+str(k),poly(k))
 for f in (np.sin,np.tanh,np.cos):report(f.__name__,np.c_[poly(2),f(x)])
 print('coef quadratic',np.linalg.lstsq(poly(2),Y,rcond=None)[0].round(3))
