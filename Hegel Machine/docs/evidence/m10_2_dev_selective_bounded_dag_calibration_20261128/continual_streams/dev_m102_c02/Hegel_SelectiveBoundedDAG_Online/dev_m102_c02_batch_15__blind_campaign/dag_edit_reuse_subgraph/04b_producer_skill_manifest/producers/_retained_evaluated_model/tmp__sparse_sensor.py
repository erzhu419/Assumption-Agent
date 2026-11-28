import numpy as np,pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_a'][k] for k in ['u','v']];A=np.c_[np.ones(len(u)),u];e=np.linalg.norm(v-A@np.linalg.lstsq(A,v,rcond=None)[0],axis=1);k=e<5*np.median(e);u,v=u[k],v[k];A=A[k];w=np.roll(u,2,axis=1)
for name in ['sinopp','sinall','tanhopp','tanhall','atanopp','atanall','ratopp','sinopp_self','sinopp_cubic','sinall_cubic']:
 def evaluate(a, details=False):
  op=np.sin if name.startswith('sin') else (np.tanh if name.startswith('tanh') else (np.arctan if name.startswith('atan') else lambda t:t/(1+t*t)))
  vals=[];bs=[]
  for j in range(4):
   d=A if name.endswith('all') or name=='sinall_cubic' else np.c_[np.ones(len(u)),u[:,j],w[:,j]]
   d=np.c_[d,op(a*(u if 'all' in name else w[:,j]))];
   if name.endswith('self'):d=np.c_[d,op(a*u[:,j])]
   if name.endswith('cubic'):d=np.c_[d,u[:,j]**3]
   b=np.linalg.lstsq(d,v[:,j],rcond=None)[0]; h=np.sum(np.linalg.qr(d)[0]**2,axis=1);vals.append((v[:,j]-d@b)/(1-h));bs.append(b)
  return (np.sqrt(np.mean(np.square(vals))),np.array(bs)) if details else np.sqrt(np.mean(np.square(vals)))
 grid=np.linspace(.1,6,120); i=np.argmin([evaluate(a) for a in grid]); opt=minimize_scalar(evaluate,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); print(name,opt.x,opt.fun); print(evaluate(opt.x,True)[1].round(5))
