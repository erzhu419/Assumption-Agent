import pickle, numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']
np.set_printoptions(precision=4,suppress=True)
print('load ranges',L.min(0),L.max(0),'large raw rows:'); print(np.column_stack((L,R))[np.argsort(np.linalg.norm(R,axis=1))[-12:]])
def bases(x):
 yield 'affine',np.c_[np.ones(len(x)),x]
 for n in (2,3,4,5):
  yield f'diagonal {n}',np.column_stack([np.ones(len(x))]+[x**k for k in range(1,n+1)])
  if n<=3:
   yield f'full {n}',np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,n+1) for j in combinations_with_replacement(range(4),k)])
 yield 'abs quadratic',np.c_[np.ones(len(x)),x,x*abs(x)]
 yield 'tanh cubic',np.c_[np.ones(len(x)),x,x**2,x**3,np.tanh(x)]
for u,v,name in [(L,R,'forward'),(R,L,'inverse')]:
 print(name)
 for tag,X in bases(u):
  B=np.linalg.lstsq(X,v,rcond=None)[0]; err=v-X@B; h=(np.linalg.qr(X)[0]**2).sum(1); e=err/(1-h[:,None]); print(tag,'LOO rmse',np.mean(e**2,0)**.5,'median/max total',np.median(abs(e.sum(1))),np.max(abs(e.sum(1))),'train',np.mean(err**2,0)**.5)
  if tag in ('diagonal 2','diagonal 3'): print(B)
