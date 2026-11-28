import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=map(s['panel_a'].get,['u','v']); b,w=m._calibration(s['panel_a']); keep=np.linalg.norm(u@w+b-v,axis=1)<2; u,v=u[keep],v[keep]
funcs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rational':lambda z:z/(1+z*z),'absrat':lambda z:z/(1+abs(z)),'softsign':lambda z:z/np.sqrt(1+z*z),'erf':__import__('scipy.special',fromlist=['erf']).erf,'cospair':lambda z:z*np.cos(z),'exponential':lambda z:z*np.exp(-z*z)}
for name,fn in funcs.items():
 def calc(k,ret=False):
  a=np.c_[np.ones(len(u)),u,fn(k*u)]
  w=m._ls(a,v); e=a@w-v
  if ret:
   h=(a*np.linalg.pinv(a).T).sum(1); return np.sqrt(np.mean((e/(1-h[:,None]))**2)),w
  return np.sqrt(np.mean(e*e))
 grid=np.linspace(.15,5,150); i=np.argmin([calc(g) for g in grid]); opt=minimize_scalar(calc,bounds=(grid[max(i-1,0)],grid[min(i+1,len(grid)-1)]),method='bounded'); cv,w=calc(opt.x,True); print(name,round(opt.x,7),round(opt.fun,8),round(cv,8));
 if name=='sin' or opt.fun<.003: print(w.round(6))
print('panel C keys',s['panel_c'].keys(),'train',len(s['train']['x']))
