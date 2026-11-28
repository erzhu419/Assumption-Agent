import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; q=t['traj']; y=t['y']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ii],axis=1) for k in range(1,d+1) for ii in combinations_with_replacement(range(x.shape[1]),k)])
def score(f,z):
 p=f[52:]@ls(f[:52],z[:52]);return np.mean(abs(p-z[52:]),axis=0)
a=s['panel_a']; u,v=a['u'],a['v']; print('times tel summary',s['eval_times'],s['telemetry_vec'],t['summary']); print('y range',y.min(),y.max()); print('traj',q.min(0),q.max(0),q.mean(0)); print('vt',np.linalg.svd(q,full_matrices=False)[2][:2]); print('center sv',np.linalg.svd(q-q.mean(0),compute_uv=False));
for mode in ['diag','full','diagpoly2','poly2','diagpoly3']:
 ps=[]
 for j in range(4):
  f=poly(u[:,j:j+1] if 'diag' in mode else u,int(mode[-1]) if 'poly' in mode else 1)
  w=least_squares(lambda w:f@w-v[:,j],ls(f,v[:,j]),loss='cauchy',f_scale=.003,max_nfev=300).x; r=abs(f@w-v[:,j]); ps.append(w); print(mode,j,w,'r',np.quantile(r,[.1,.5,.8,.9,.95,1]))
 if mode=='diag':
  off,gain=np.array(ps).T; z=(x-off)/gain
 if mode=='full':zz=(x-np.array(ps)[:,0])@np.linalg.inv(np.array(ps)[:,1:].T)
b=s['panel_b']; B=ls(aff(b['u']),b['v']); print('B singular',np.linalg.svd(B[1:],compute_uv=False)); c=s['panel_c'];
for k,vv in [('v1',c['v1']),('v2',c['v2'])]:
 uu=c['u']; print(k,'range',vv.min(0),vv.max(0));
 for name,fun,inv in [('identity',lambda x:x,lambda x:x),('tanh',np.tanh,np.arctanh),('sin',np.sin,np.arcsin),('asinh',np.arcsinh,np.sinh),('atan',np.arctan,np.tan),('softsign',lambda x:x/(1+abs(x)),lambda x:x/(1-abs(x)))]:
  rr=[]
  for target in vv.T:
   X=aff(uu); w=ls(X,target); fit=least_squares(lambda b: b[-2]+b[-1]*fun(X@b[:-2])-target,np.r_[w,0,1],max_nfev=350); rr.append(np.sqrt(np.mean(fit.fun**2)))
  print(name,rr)
 print('poly2',np.sqrt(np.mean((poly(uu,2)@ls(poly(uu,2),vv)-vv)**2,0)))
for k,w in [('raw',x),('diag',z),('full',zz),('br',aff(x)@B),('bd',aff(z)@B),('bf',aff(zz)@B)]:
 for d in range(1,4):
  print(k,d,score(poly(w,d),y),score(poly(w,d),q).mean());
 for name,f in [('tanh',np.tanh),('atan',np.arctan),('sin',np.sin),('asinh',np.arcsinh)]:
  print(k,name,[(g,round(score(np.c_[aff(w),f(g*w)],y),4),round(score(np.c_[aff(w),f(g*w)],q).mean(),4)) for g in [.5,1,2,4]])
