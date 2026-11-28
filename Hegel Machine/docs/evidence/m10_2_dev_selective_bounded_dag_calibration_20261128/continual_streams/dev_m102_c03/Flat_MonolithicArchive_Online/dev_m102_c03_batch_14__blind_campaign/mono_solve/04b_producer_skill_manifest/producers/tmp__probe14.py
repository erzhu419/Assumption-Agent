import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def score(a,b):
 q,r=np.linalg.qr(a,mode='reduced'); d=b-q@(q.T@b); return np.sqrt(np.mean((d/(1-(q*q).sum(1))[:,None])**2))
def aff(u):return np.c_[np.ones(len(u)),u]
def pairs(u):return np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i)])
for k in ['v1','v2']:
 u=s['panel_c']['u']; v=s['panel_c'][k]; print(k)
 funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'erf':__import__('scipy').special.erf,'softsin':lambda x:x/(1+x*x),'softsqrt':lambda x:x/np.sqrt(1+x*x),'lorentz':lambda x:1/(1+x*x),'softplus':lambda x:np.logaddexp(0,x),'gauss':lambda x:np.exp(-x*x)}
 choices=[]
 for name,f in funcs.items():
  for mode in [0,1,2,3]:
   def mat(w):
    z=f(u*w); return [aff(z),np.c_[aff(u),z],np.c_[aff(u),pairs(u),z],np.c_[aff(z),pairs(u)]][mode]
   res=minimize_scalar(lambda w:score(mat(w),v),bounds=(.15,4),method='bounded'); choices.append((res.fun,name,mode,res.x))
 print(sorted(choices)[:12]); print('poly',[(i,score(np.c_[aff(u),*(u**j for j in range(2,i+1))],v)) for i in range(1,5)]); print('quad',score(np.c_[aff(u),u*u,pairs(u)],v))
t=s['eval_times']; T=s['train']['traj']; _,a,q=np.linalg.svd(T,full_matrices=False); print('time sv',a[:4]); print('modes',q[:2]); print('time proj',q[:2].T@q[:2]); print('test bounds',s['test']['x'].min(0),s['test']['x'].max(0))
