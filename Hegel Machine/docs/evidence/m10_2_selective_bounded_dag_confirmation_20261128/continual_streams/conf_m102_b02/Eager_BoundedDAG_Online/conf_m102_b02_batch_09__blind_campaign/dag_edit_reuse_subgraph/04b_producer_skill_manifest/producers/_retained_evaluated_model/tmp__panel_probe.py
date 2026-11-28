import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u=s['panel_a']['u']; v=s['panel_a']['v']; a=np.c_[np.ones(len(u)),u]; fit=np.linalg.lstsq(a,v,rcond=None)[0]; k=s['telemetry_vec'][1]
print('u/v/train/test ranges',[(x.min(0),x.max(0)) for x in (u,v,s['train']['x'],s['test']['x'])]); print('affine',fit); print('affine errors',np.round(np.linalg.norm(a@fit-v,axis=1),4)); print('telemetry',s['telemetry_vec'])
funcs={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'cos':np.cos,'erf':__import__('scipy.special',fromlist=['erf']).erf,'square':lambda x:x*x,'cube':lambda x:x**3,'abs':np.abs,'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'softplus':lambda x:np.logaddexp(x,0)}
for name,f in funcs.items():
 h=f(k*u); errs=[]
 for j in range(4):
  opts=[]
  for i in range(4):
   q=np.c_[a,h[:,i]]; c=np.linalg.lstsq(q,v[:,j],rcond=None)[0]; opts.append((np.linalg.norm(q@c-v[:,j]),i,round(c[-1],5)))
  errs.append(min(opts))
 q=np.c_[a,h]; c=np.linalg.lstsq(q,v,rcond=None)[0]; print(name,'single', [(round(e,5),i,c) for e,i,c in errs], 'mixed',round(np.linalg.norm(q@c-v),5))
for name,f in funcs.items():
 scores=[]
 for j in range(4):
  l=fit[:,j]; y=v[:,j];
  def r(p):
   t=a@p[:5]; return t+p[5]*f(k*t)-y
  best=min((least_squares(r,np.r_[l,z],max_nfev=250) for z in [-.5,0,.5]),key=lambda t:t.cost)
  scores.append((round(np.linalg.norm(best.fun),6),round(best.x[-1],5)))
 print('after affine',name,scores)
