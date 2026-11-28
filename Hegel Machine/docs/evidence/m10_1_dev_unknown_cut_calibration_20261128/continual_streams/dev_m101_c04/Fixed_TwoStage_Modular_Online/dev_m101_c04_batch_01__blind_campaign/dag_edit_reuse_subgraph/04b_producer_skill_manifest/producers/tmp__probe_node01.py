import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
d=node_00.run_node({'__input__':s}); x,y,t=d['z_tr'],d['y'],d['traj']; a,b=node_00._calibration(s)[0:2]; u=(s['train']['x']-a)/b
np.set_printoptions(precision=4,suppress=True)
print('time',d['eval_times'],'z ranges',x.min(0),x.max(0),'trajectory svals',np.linalg.svd(t,compute_uv=False))
def poly(x,k):
 return np.column_stack([np.prod(x[:,idx],axis=1) for j in range(k+1) for idx in itertools.combinations_with_replacement(range(x.shape[1]),j)])
def score(A,Y):
 f=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0]; r=A@f-Y
 return [round(np.abs(v).mean(),6) for v in (r[:52],r[52:])]
for w in (u,x):
 for k in (1,2,3):
  print('poly',k,score(poly(w,k),y),score(poly(w,k),t))
 for k in (.5,1.,1.5,2.,3.):
  print('trig',k,score(np.column_stack((poly(w,2),np.sin(k*w),np.cos(k*w))),y),score(np.column_stack((poly(w,2),np.sin(k*w),np.cos(k*w))),t))
print('traj rows',t[:7]); print('trajectory endpoint fits by horizon:')
for j in range(t.shape[1]): print(j,score(poly(x,2),t[:,j]),score(poly(x,3),t[:,j]))
for k in (1,2):
 A=poly(x,k); f=np.linalg.lstsq(A[:52],y[:52],rcond=None)[0]; print('scalar coefficients',k,f)
