import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
np.set_printoptions(precision=5,suppress=True,linewidth=140)
print('ranges z,u,T,y:',[(w.min(),w.max()) for w in (z,u,T,y)]); print('trajectory singular values',np.linalg.svd(T,compute_uv=False)); print('trajectory first rows',T[:5]); print('panel norms',np.std(v,axis=0)); print('panel linear',np.linalg.lstsq(u,v,rcond=None)[0].T)
def poly(x,degree):
 return np.column_stack([np.prod(x[:,c],axis=1) for d in range(degree+1) for c in combinations_with_replacement(range(x.shape[1]),d)])
def score(F,Y,split,k=0):
 x=F[:split]; b=Y[:split]; w=np.linalg.lstsq(x,b,rcond=None)[0] if not k else np.linalg.solve(x.T@x+k*np.eye(x.shape[1]),x.T@b)
 return np.mean(abs(F[split:]@w-Y[split:])),np.sqrt(np.mean((x@w-b)**2))
for d in range(1,5):
 for k in (0,.03,.3,1):
  print('poly',d,k,'panel',score(poly(u,d),v,18,k),'Y',score(poly(z,d),y,52,k),'T',score(poly(z,d),T,52,k))
for fn in (np.tanh,np.sin,np.sinh):
 F=lambda x:np.column_stack((np.ones(len(x)),x,fn(x),x**2,x*np.roll(x,-1,axis=1)))
 print(fn.__name__,'panel',score(F(u),v,18,.001),'Y',score(F(z),y,52,.001),'T',score(F(z),T,52,.001))
for degree in (1,2,3):
 W=np.linalg.lstsq(poly(z,degree),np.column_stack((y,T)),rcond=None)[0]; R=poly(z,degree)@W-np.column_stack((y,T)); print('residual std by horizon',degree,np.std(R,axis=0)); print('linear coefficients or scalar poly coefficients',W if degree==1 else W[:,0])
print('panel linear residuals first eight', (v-u@np.linalg.lstsq(u,v,rcond=None)[0])[:8])
